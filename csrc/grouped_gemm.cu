#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"


#ifndef GROUPED_GEMM_DEVICE_CAPABILITY
#error "Undefined compute capability"
#endif


#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
  GROUPED_GEMM_STRINGIFY_HELPER(x)


#if GROUPED_GEMM_DEVICE_CAPABILITY >= 90
// TODO(one): Update this for SM90 when it's supported by CUTLASS.
#define GROUPED_GEMM_DEVICE_TAG ::cutlass::arch::Sm80
#elif GROUPED_GEMM_DEVICE_CAPABILITY >= 80
#define GROUPED_GEMM_DEVICE_TAG ::cutlass::arch::Sm80
#else
#error "Unsupported compute capability " GROUPED_GEMM_STRINGIFY(GROUPED_GEMM_DEVICE_CAPABILITY)
#endif

template <typename T>
torch::Tensor copy_to_device(const std::vector<T> &x, const torch::Device &device) {
    size_t bytes = x.size() * sizeof(T);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
    torch::Tensor out = torch::empty(bytes, options);

    cudaError_t status = cudaMemcpyAsync(out.data_ptr(),
                                         x.data(), bytes,
                                         cudaMemcpyHostToDevice,
                                         c10::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    return out;
}


template <typename T>
static void reorder_array(T* data, const std::vector<size_t>& indices) {
    // For now, simply create a copy of the data and then copy over to the original.
    std::vector<T> copy(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        copy.at(i) = data[indices[i]];
    }

    memcpy(data, copy.data(), indices.size() * sizeof(T));
}

constexpr int kDynamicDim = -1;

template <typename T>
torch::Tensor typed_empty(size_t numel, const torch::Device& device) {
    return torch::empty(numel * sizeof(T), torch::dtype(torch::kInt8).device(device));
}

struct GemmProblem {
    cutlass::gemm::GemmCoord dims;
    int lda, ldb, ldc;
    // All offsets are in elements.
    int a_offset, b_offset, c_offset;
};

struct ExtractGemmProblemK {
    __device__ ::cuda::std::tuple<int&> operator()(GemmProblem& problem) const {
        return {problem.dims.k()};
    }
};

template <
    // If `k` is dynamic, we sort the problems by `k` in descending order.
    // Otherwise, `m` is dynamic, and no sorting happens.
    bool dynamic_k,
    typename ElementA, typename ElementB, typename ElementC,
    typename LayoutA, typename LayoutB, typename LayoutC,
    typename Args
>
__global__ void FillArguments(
    int num_experts, const int64_t* batch_sizes,
    ElementA* ptr_a, ElementB* ptr_b, ElementC* ptr_c,
    Args args, cutlass::gemm::GemmCoord dims
) {
    const int expert_idx = threadIdx.x;
    const int batch_size = expert_idx < num_experts ? batch_sizes[expert_idx] : -1;

    if (dynamic_k) {
        assert(dims.k() == kDynamicDim);
        dims.k() = batch_size;
    } else {
        assert(dims.m() == kDynamicDim);
        dims.m() = batch_size;
    }

    using BlockScan = cub::BlockScan<int, at::cuda::detail::CUDA_NUM_THREADS>;
    using BlockSort = cub::BlockRadixSort<GemmProblem, at::cuda::detail::CUDA_NUM_THREADS, 1>;

    union SharedMemory {
        BlockScan::TempStorage scan_storage;
        BlockSort::TempStorage sort_storage;
    };
    __shared__ SharedMemory shared_memory;

    int dynamic_dim = dynamic_k ? dims.k() : dims.m();
    int dynamic_dim_cumsum;
    BlockScan(shared_memory.scan_storage).ExclusiveSum(dynamic_dim, dynamic_dim_cumsum);
    __syncthreads();

    GemmProblem problem[1] = {
        GemmProblem {
            .dims = dims,
            .lda = LayoutA::packed({dims.m(), dims.k()}).stride(0),
            .ldb = LayoutB::packed({dims.k(), dims.n()}).stride(0),
            .ldc = LayoutC::packed({dims.m(), dims.n()}).stride(0),
            .a_offset = dynamic_k
                ? (dims.m() * dynamic_dim_cumsum)
                : (dynamic_dim_cumsum * dims.k()),
            .b_offset = (dynamic_k ? dynamic_dim_cumsum : expert_idx * dims.k()) * dims.n(),
            .c_offset = (dynamic_k ? expert_idx * dims.m() : dynamic_dim_cumsum) * dims.n(),
        },
    };

    if constexpr (dynamic_k) {
        BlockSort(shared_memory.sort_storage).SortDescending(problem, ExtractGemmProblemK{});
        // Quoting the CUB documentation (https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockRadixSort.html):
        // > A subsequent __syncthreads() threadblock barrier should be invoked after calling this method if the collective’s temporary storage [...]
        // > is **to be reused or repurposed**.
        // We don't need `__syncthreads()` here, since we don't do either of these things.
    }

    if (expert_idx < num_experts) {
        args.problem_sizes[expert_idx] = problem[0].dims;
        args.lda[expert_idx] = problem[0].lda;
        args.ldb[expert_idx] = problem[0].ldb;
        args.ldc[expert_idx] = problem[0].ldc;

        args.ptr_A[expert_idx] = ptr_a + problem[0].a_offset;
        args.ptr_B[expert_idx] = ptr_b + problem[0].b_offset;
        args.ptr_C[expert_idx] = ptr_c + problem[0].c_offset;
    }
}

template <
    typename layoutA,
    typename layoutB,
    typename operandDtype,
    typename accumulatorDtype,
    bool dynamic_k,

    // default config
    typename OperatorClass_ = ::cutlass::arch::OpClassTensorOp,
    typename DefaultConfig_ = ::cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass_, GROUPED_GEMM_DEVICE_TAG, operandDtype, operandDtype, operandDtype, accumulatorDtype>,
    
    int kAlignmentA_ = DefaultConfig_::kAlignmentA,
    int kAlignmentB_ = DefaultConfig_::kAlignmentB,
    typename ThreadblockShape_ = typename DefaultConfig_::ThreadblockShape,
    typename WarpShape_ = typename DefaultConfig_::WarpShape,
    typename InstructionShape_ = typename DefaultConfig_::InstructionShape,
    typename EpilogueOutputOp_ = typename DefaultConfig_::EpilogueOutputOp,
    int kStages_ = DefaultConfig_::kStages>
class cutlass_grouped_gemm {
public:
    using GroupedGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        // A operand.
        operandDtype,
        layoutA,
        ::cutlass::ComplexTransform::kNone,
        kAlignmentA_,
        // B operand.
        operandDtype,
        layoutB,
        ::cutlass::ComplexTransform::kNone,
        kAlignmentB_,
        // C operand.
        operandDtype,
        ::cutlass::layout::RowMajor,
        accumulatorDtype,
        OperatorClass_,
        GROUPED_GEMM_DEVICE_TAG,
        ThreadblockShape_,
        WarpShape_,
        InstructionShape_,
        EpilogueOutputOp_,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
        // This parameter is passed in at present to match the APIs of other kernels. The parameter
        // is unused within the kernel.
        ::cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        // TODO(tgale): Experiment with GroupScheduleMode.
        // TODO(tgale): Tune this for SM90.
        kStages_>::GemmKernel;

    using GemmGrouped = ::cutlass::gemm::device::GemmGrouped<GroupedGemmKernel>;


    static torch::Tensor run(
                       torch::Tensor a,
				       torch::Tensor b,
				       torch::Tensor c,
				       torch::Tensor batch_sizes,
                       const cutlass::gemm::GemmCoord& coord) {
        using LayoutA = typename GemmGrouped::LayoutA;
        using LayoutB = typename GemmGrouped::LayoutB;
        using LayoutC = typename GemmGrouped::LayoutC;

        using ElementA = typename GemmGrouped::ElementA;
        using ElementB = typename GemmGrouped::ElementB;
        using ElementC = typename GemmGrouped::ElementC;

        int64_t num_experts = batch_sizes.size(0);

        torch::Tensor lda, ldb, ldc, ptr_a, ptr_b, ptr_c, problem_sizes;
        int threadblock_count{};
        std::vector<cutlass::gemm::GemmCoord> problem_sizes_host;
        if (batch_sizes.is_cuda()) {
            TORCH_CHECK(
                num_experts <= at::cuda::detail::CUDA_NUM_THREADS,
                "At most ", at::cuda::detail::CUDA_NUM_THREADS,
                " experts are supported when batch_sizes is a CUDA tensor, but got ", num_experts
            );

            lda = typed_empty<int64_t>(num_experts, a.device());
            ldb = typed_empty<int64_t>(num_experts, a.device());
            ldc = typed_empty<int64_t>(num_experts, a.device());
            ptr_a = typed_empty<ElementA*>(num_experts, a.device());
            ptr_b = typed_empty<ElementB*>(num_experts, a.device());
            ptr_c = typed_empty<ElementC*>(num_experts, a.device());
            problem_sizes = typed_empty<cutlass::gemm::GemmCoord>(num_experts, a.device());

            // We don't know the real number number of tiles, so we just base the count on occupancy here.
            threadblock_count = GemmGrouped::sufficient();
        } else {
            problem_sizes_host.resize(num_experts);

            // Create the host arrays of leading dimension data and pointer data.
            std::vector<int64_t> lda_host(num_experts), offsets_a(num_experts);
            std::vector<int64_t> ldb_host(num_experts), offsets_b(num_experts);
            std::vector<int64_t> ldc_host(num_experts), offsets_c(num_experts);
            int64_t elements_a = 0, elements_b = 0, elements_c = 0;
            std::vector<ElementA *> ptr_a_host(num_experts);
            std::vector<ElementB *> ptr_b_host(num_experts);
            std::vector<ElementC *> ptr_c_host(num_experts);

            for (int i = 0; i < num_experts; ++i) {
                auto& problem = problem_sizes_host[i];
                problem = coord;
                (dynamic_k ? problem.k() : problem.m()) = batch_sizes.data_ptr<int64_t>()[i];

                lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
                ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
                ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);

                offsets_a[i] = elements_a;
                offsets_b[i] = elements_b;
                offsets_c[i] = elements_c;

                ptr_a_host[i] = (ElementA*)a.data_ptr() + offsets_a[i];
                ptr_b_host[i] = (ElementB*)b.data_ptr() + offsets_b[i];
                ptr_c_host[i] = (ElementC*)c.data_ptr() + offsets_c[i];

                elements_a += problem.m() * problem.k();
                elements_b += problem.k() * problem.n();
                elements_c += problem.m() * problem.n();
            }

            // Sort problems
            if (dynamic_k) {
                std::vector<size_t> indices(num_experts);
                std::iota(indices.begin(), indices.end(), 0);
                std::stable_sort(indices.begin(), indices.end(), [&problem_sizes_host](size_t i, size_t j) {
                    return problem_sizes_host[i].k() > problem_sizes_host[j].k();
                });

                reorder_array(problem_sizes_host.data(), indices);
                reorder_array(lda_host.data(), indices);
                reorder_array(ldb_host.data(), indices);
                reorder_array(ldc_host.data(), indices);
                reorder_array(ptr_a_host.data(), indices);
                reorder_array(ptr_b_host.data(), indices);
                reorder_array(ptr_c_host.data(), indices);
            }

            // Copy the problem sizes, pointers and leading dimension data to the device.
            lda = copy_to_device(lda_host, a.device());
            ldb = copy_to_device(ldb_host, a.device());
            ldc = copy_to_device(ldc_host, a.device());
            ptr_a = copy_to_device(ptr_a_host, a.device());
            ptr_b = copy_to_device(ptr_b_host, a.device());
            ptr_c = copy_to_device(ptr_c_host, a.device());
            problem_sizes = copy_to_device(problem_sizes_host, a.device());

            threadblock_count = GemmGrouped::sufficient(problem_sizes_host.data(), num_experts);
        }

        if (!threadblock_count) {
            TORCH_CHECK(false, "Grouped GEMM execution not possible with HW");
        }

        // We don't specify `host_problem_sizes` because we don't know them.
        // This is fine, since the default `GroupScheduleMode_` is `kDeviceOnly`.
        typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(/*alpha=*/1.0f, /*beta=*/0.0f);
        typename GemmGrouped::Arguments arguments((cutlass::gemm::GemmCoord*)problem_sizes.data_ptr(),
                            (int)num_experts,
                            (int)threadblock_count,
                            epilogue_op,
                            (ElementA**)ptr_a.data_ptr(),
                            (ElementB**)ptr_b.data_ptr(),
                            (ElementC**)ptr_c.data_ptr(),
                            (ElementC**)ptr_c.data_ptr(),
                            /*lda=*/(int64_t*)lda.data_ptr(),
                            /*ldb=*/(int64_t*)ldb.data_ptr(),
                            /*ldc=*/(int64_t*)ldc.data_ptr(),
                            /*ldd=*/(int64_t*)ldc.data_ptr(),
                            (cutlass::gemm::GemmCoord*)(batch_sizes.is_cuda() ? nullptr : problem_sizes_host.data()));

        if (batch_sizes.is_cuda()) {
            // Use a single block so that we don't need any additional global memory.
            FillArguments<
                dynamic_k,
                ElementA, ElementB, ElementC,
                LayoutA, LayoutB, LayoutC
            ><<<1, at::cuda::detail::CUDA_NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
                num_experts, batch_sizes.data_ptr<int64_t>(),
                (ElementA*)a.data_ptr(), (ElementB*)b.data_ptr(), (ElementC*)c.data_ptr(),
                arguments, coord
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // Run Grouped GEMM
        GemmGrouped gemm;

        int64_t workspace_size = gemm.get_workspace_size(arguments);
        torch::Tensor workspace = typed_empty<uint8_t>(workspace_size, a.device());

        // Initialize the kernel.
        if(gemm.initialize(arguments, workspace.data_ptr()) != cutlass::Status::kSuccess) {
            TORCH_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
        }

        // Execute the kernel in the current stream.
        if(gemm.run(c10::cuda::getCurrentCUDAStream()) != cutlass::Status::kSuccess) {
            TORCH_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
        }
        return c;
    }
};


namespace grouped_gemm {
    // NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
    // assumed to be batched with fixed sized batches.
    //
    // TODO(tgale): Validate alignment is true for every batch element.
    void GroupedGemm(torch::Tensor a,
            torch::Tensor b,
            torch::Tensor c,
            torch::Tensor batch_sizes,
            bool trans_a, bool trans_b) {
        // a: (tokens, hidden_in)

        // When [trans_a = False]
        // b - trans_b=false: (num_experts, hidden_in, hidden_out)
        // b - trans_b=true:  (num_experts, hidden_out, hidden_in)
        // c: (tokens, hidden_out)

        // When [trans_a = True]
        // b: (tokens, hidden_out)
        // c: (num_experts, hidden_in, hidden_out)

        // Check function arguments
        // NOTE: We only support 'trans_a' or 'trans_b', not both.
        TORCH_CHECK(!(trans_a && trans_b));

        // Check tensor dtype, device and arrangement
        // NOTE: We support transposition through the 'trans_b' flag.
        TORCH_CHECK(a.is_contiguous());
        TORCH_CHECK(b.is_contiguous());
        TORCH_CHECK(c.is_contiguous()); 

        TORCH_CHECK(a.is_cuda());
        TORCH_CHECK(b.is_cuda());
        TORCH_CHECK(c.is_cuda());
        TORCH_CHECK(batch_sizes.is_cuda() || batch_sizes.is_cpu());

        TORCH_CHECK(a.scalar_type() == torch::kBFloat16);
        TORCH_CHECK(b.scalar_type() == torch::kBFloat16);
        TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
        TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

        // Check tensor shapes
        TORCH_CHECK(batch_sizes.ndimension() == 1);
        TORCH_CHECK(a.ndimension() == 2);

        const size_t tokens = a.size(0), hidden_in = a.size(1);
        const size_t num_experts = batch_sizes.size(0);
        size_t hidden_out;
        if (trans_a) {
            TORCH_CHECK(b.ndimension() == 2);
            TORCH_CHECK(c.ndimension() == 3);

            hidden_out = b.size(1);
            TORCH_CHECK(b.size(0) == tokens);
            TORCH_CHECK(c.size(0) == num_experts);
            TORCH_CHECK(c.size(1) == hidden_in);
            TORCH_CHECK(c.size(2) == hidden_out);
        }
        else {
            // We expected a CUDA tensor with three dimensions and shape
            //  for 'b'.
            TORCH_CHECK(b.ndimension() == 3);
            TORCH_CHECK(c.ndimension() == 2);

            // Validate the contraction dimensions match.
            const size_t b_hidden_in = trans_b ? b.size(2) : b.size(1);
            hidden_out = trans_b ? b.size(1) : b.size(2);

            TORCH_CHECK(b.size(0) == num_experts);
            TORCH_CHECK(b_hidden_in == hidden_in);

            // Validate the output shape
            TORCH_CHECK(c.size(0) == tokens);
            TORCH_CHECK(c.size(1) == hidden_out);
        }

        if (trans_a) {
            // Only sort problems when trans_a = True because only this case K are different
            const auto coord = cutlass::gemm::GemmCoord(hidden_in, hidden_out, kDynamicDim);
            ::cutlass_grouped_gemm<::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor, ::cutlass::bfloat16_t, float, true>::run(
                a, b, c, batch_sizes, coord
            );
        }
        else if (trans_b) {
            const auto coord = cutlass::gemm::GemmCoord(kDynamicDim, hidden_out, hidden_in);
            ::cutlass_grouped_gemm<::cutlass::layout::RowMajor, ::cutlass::layout::ColumnMajor, ::cutlass::bfloat16_t, float, false>::run(
                a, b, c, batch_sizes, coord
            );
        }
        else {
            const auto coord = cutlass::gemm::GemmCoord(kDynamicDim, hidden_out, hidden_in);
            ::cutlass_grouped_gemm<::cutlass::layout::RowMajor, ::cutlass::layout::RowMajor, ::cutlass::bfloat16_t, float, false>::run(
                a, b, c, batch_sizes, coord
            );
        }
    }
};
