// 삼각형 래스터라이저 (MIT).
//
// 두 가지 모드를 제공한다.
//   1) uv_coverage   : UV 아틀라스용 2D 래스터화. 깊이 없음.
//                      (nvdiffrast 의 rasterize 중 to_glb 가 쓰는 부분 대체)
//   2) mesh_coverage : 화면 공간 3D 래스터화. z 버퍼 포함.
//                      (미리보기 렌더용)
//
// 두 모드 모두 커버리지(삼각형 id)만 구하고, 바리센트릭은 파이썬에서 계산한다.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------- 2D (UV)
__global__ void uv_coverage_kernel(
    const float* __restrict__ uv,     // [V,2], UV 공간 [0,1]
    const int*   __restrict__ faces,  // [F,3]
    const int F, const int H, const int W,
    int* __restrict__ tri_buf)        // [H,W], 0 초기화
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    const int i0 = faces[3*f+0], i1 = faces[3*f+1], i2 = faces[3*f+2];
    const float x0 = uv[2*i0] * W, y0 = uv[2*i0+1] * H;
    const float x1 = uv[2*i1] * W, y1 = uv[2*i1+1] * H;
    const float x2 = uv[2*i2] * W, y2 = uv[2*i2+1] * H;

    const float area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
    if (area == 0.0f) return;
    const float inv = 1.0f / area;

    int px0 = max((int)floorf(fminf(x0, fminf(x1,x2)) - 0.5f), 0);
    int px1 = min((int)ceilf (fmaxf(x0, fmaxf(x1,x2)) + 0.5f), W-1);
    int py0 = max((int)floorf(fminf(y0, fminf(y1,y2)) - 0.5f), 0);
    int py1 = min((int)ceilf (fmaxf(y0, fmaxf(y1,y2)) + 0.5f), H-1);

    for (int j = py0; j <= py1; ++j) {
        const float py = j + 0.5f;
        for (int i = px0; i <= px1; ++i) {
            const float px = i + 0.5f;
            const float l0 = ((x1-px)*(y2-py) - (x2-px)*(y1-py)) * inv;
            if (l0 < 0.0f) continue;
            const float l1 = ((x2-px)*(y0-py) - (x0-px)*(y2-py)) * inv;
            if (l1 < 0.0f) continue;
            if (1.0f - l0 - l1 < 0.0f) continue;
            atomicMax(&tri_buf[j*W + i], f + 1);   // 겹치면 인덱스 큰 쪽 (결정적)
        }
    }
}

// ------------------------------------------------------------- 3D (screen)
// key = (z_bits << 32) | tri_id  에 atomicMin -> 가장 가까운 삼각형이 이긴다.
__global__ void mesh_coverage_kernel(
    const float* __restrict__ xyz,    // [V,3] 화면공간: x∈[0,W], y∈[0,H], z>=0 (작을수록 가까움)
    const int*   __restrict__ faces,  // [F,3]
    const int F, const int H, const int W,
    unsigned long long* __restrict__ key_buf)   // [H,W], 0xFFFF.. 초기화
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    const int i0 = faces[3*f+0], i1 = faces[3*f+1], i2 = faces[3*f+2];
    const float x0 = xyz[3*i0], y0 = xyz[3*i0+1], z0 = xyz[3*i0+2];
    const float x1 = xyz[3*i1], y1 = xyz[3*i1+1], z1 = xyz[3*i1+2];
    const float x2 = xyz[3*i2], y2 = xyz[3*i2+1], z2 = xyz[3*i2+2];

    const float area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
    if (area == 0.0f) return;
    const float inv = 1.0f / area;

    int px0 = max((int)floorf(fminf(x0, fminf(x1,x2)) - 0.5f), 0);
    int px1 = min((int)ceilf (fmaxf(x0, fmaxf(x1,x2)) + 0.5f), W-1);
    int py0 = max((int)floorf(fminf(y0, fminf(y1,y2)) - 0.5f), 0);
    int py1 = min((int)ceilf (fmaxf(y0, fmaxf(y1,y2)) + 0.5f), H-1);

    for (int j = py0; j <= py1; ++j) {
        const float py = j + 0.5f;
        for (int i = px0; i <= px1; ++i) {
            const float px = i + 0.5f;
            const float l0 = ((x1-px)*(y2-py) - (x2-px)*(y1-py)) * inv;
            if (l0 < 0.0f) continue;
            const float l1 = ((x2-px)*(y0-py) - (x0-px)*(y2-py)) * inv;
            if (l1 < 0.0f) continue;
            const float l2 = 1.0f - l0 - l1;
            if (l2 < 0.0f) continue;

            const float z = fmaxf(l0*z0 + l1*z1 + l2*z2, 0.0f);
            const unsigned long long key =
                ((unsigned long long)__float_as_uint(z) << 32) | (unsigned int)(f + 1);
            atomicMin(&key_buf[j*W + i], key);
        }
    }
}

torch::Tensor uv_coverage(torch::Tensor uv, torch::Tensor faces, int64_t H, int64_t W)
{
    TORCH_CHECK(uv.is_cuda() && faces.is_cuda(), "uv/faces 는 CUDA 텐서여야 합니다");
    uv = uv.contiguous().to(torch::kFloat32);
    faces = faces.contiguous().to(torch::kInt32);
    auto tri = torch::zeros({H, W}, torch::TensorOptions().dtype(torch::kInt32).device(uv.device()));
    const int F = faces.size(0);
    if (F == 0) return tri;
    const int threads = 256, blocks = (F + threads - 1) / threads;
    uv_coverage_kernel<<<blocks, threads>>>(
        uv.data_ptr<float>(), faces.data_ptr<int>(), F, (int)H, (int)W, tri.data_ptr<int>());
    // C10_CUDA_CHECK 는 torch 빌드에 따라 없는 심볼을 참조하는 경우가 있어
    // 표준 TORCH_CHECK 로 확인한다.
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "래스터 커널 실패: ", cudaGetErrorString(err));
    }
    return tri;
}

torch::Tensor mesh_coverage(torch::Tensor xyz, torch::Tensor faces, int64_t H, int64_t W)
{
    TORCH_CHECK(xyz.is_cuda() && faces.is_cuda(), "xyz/faces 는 CUDA 텐서여야 합니다");
    xyz = xyz.contiguous().to(torch::kFloat32);
    faces = faces.contiguous().to(torch::kInt32);
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(xyz.device());
    auto keys = torch::full({H, W}, (int64_t)-1, opts);   // 비트 전부 1 = 최대값
    const int F = faces.size(0);
    if (F == 0) return keys;
    const int threads = 256, blocks = (F + threads - 1) / threads;
    mesh_coverage_kernel<<<blocks, threads>>>(
        xyz.data_ptr<float>(), faces.data_ptr<int>(), F, (int)H, (int)W,
        reinterpret_cast<unsigned long long*>(keys.data_ptr<int64_t>()));
    // C10_CUDA_CHECK 는 torch 빌드에 따라 없는 심볼을 참조하는 경우가 있어
    // 표준 TORCH_CHECK 로 확인한다.
    {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "래스터 커널 실패: ", cudaGetErrorString(err));
    }
    return keys;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uv_coverage",   &uv_coverage,   "UV 공간 2D 커버리지");
    m.def("mesh_coverage", &mesh_coverage, "화면 공간 3D 커버리지 (z 버퍼)");
}
