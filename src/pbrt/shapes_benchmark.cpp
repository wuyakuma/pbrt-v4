#include <benchmark/benchmark.h>

#include "pbrt/pbrt.h"
#include "pbrt/util/mesh.h"
#include "pbrt/util/rng.h"
#include "pbrt/util/transform.h"
#include "pbrt/util/vecmath.h"
#include "pbrt/util/sampling.h"
#include "pbrt/shapes.h"
#include "pbrt/interaction.h"

using namespace pbrt;

struct PBRTInit {
  private:
    PBRTInit() { InitBufferCaches(); };

  public:
    static PBRTInit& Init() {
        static PBRTInit Instance;
        return Instance;
    }
};

struct FRandomBounds {
    std::vector<Bounds3f> Bounds;

    FRandomBounds(int Count, const Bounds3f& Range) {
        PBRTInit::Init();

        Bounds.resize(Count);
        RNG rng;

        for (int i = 0; i < Count; ++i) {
            Bounds[i] =
                Bounds3f(Point3f(Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                                 Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                                 Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x)),
                         Point3f(Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                                 Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                                 Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x)));
        }
    }
};

struct FRandomTriangle {
    pstd::vector<Shape> Shapes;
    Transform IdentityTransform;
    Allocator alloc;

    FRandomTriangle(int Count, const Bounds3f& Range) {
        PBRTInit::Init();
        Triangle::Init(alloc);

        RNG rng;

        std::vector<Point3f> Positions;
        std::vector<Vector3f> TangentXs;
        std::vector<Normal3f> TangentZs;
        std::vector<Point2f> UVs;
        std::vector<int> Indices;

        Positions.resize(Count * 3);
        TangentXs.resize(Count * 3);
        TangentZs.resize(Count * 3);
        UVs.resize(Count * 3);
        Indices.resize(Count * 3);

        for (int i = 0; i < Count; ++i) {
            const Point3f P[3] = {
                Point3f(Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                        Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                        Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x)),
                Point3f(Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                        Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                        Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x)),
                Point3f(Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                        Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                        Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x)),
            };

            Vector3f Z = Normalize(Cross(P[2] - P[0], P[1] - P[0]));

            Vector3f S[3];

            Normal3f N[3]{Normal3f(Z), Normal3f(Z), Normal3f(Z)};

            Point2f UV[3];

            Vector3f X, Y;
            CoordinateSystem(Z, &X, &Y);

            //SquareMatrix<4> Matrix(X, Y, Z, Vector3f(0, 0, 0));
            Transform Project = LookAt(P[0], P[0] + Z, Y);

            for (int i = 0; i < 3; ++i) {
                Point3f Pt = Project(Point3f(P[i] - P[0]));

                UV[i].x = Pt.x;
                UV[i].y = Pt.y;

                S[i] = X;
            }

            for (int j = 0; j < 3; ++j) {
                Indices[i * 3 + j] = i * 3 + j;

                Positions[i * 3 + j] = P[j];
                TangentXs[i * 3 + j] = S[j];
                TangentZs[i * 3 + j] = N[j];
                UVs[i * 3 + j] = UV[j];
            }
        }

        TriangleMesh* mesh = alloc.new_object<TriangleMesh>(
            IdentityTransform, false, Indices, Positions, std::vector<Vector3f>(),
            TangentZs, UVs, std::vector<int>(), alloc);
        Shapes = Triangle::CreateTriangles(mesh, alloc);
    }
};

struct FRandomRays {
    std::vector<Ray> Rays;

    struct FInvDir {
        Vector3f InvDir;
        int DirIsNeg[3];
    };
    std::vector<FInvDir> InvDirs;

    FRandomRays(int Count, const Bounds3f& Range) {
        PBRTInit::Init();

        Rays.resize(Count);
        InvDirs.resize(Count);
        RNG rng;

        for (int i = 0; i < Count; ++i) {
            const Vector3f Dir = SampleUniformSphere(Point2f(rng.Uniform<float>(), rng.Uniform<float>()));
            Rays[i] = Ray(Point3f(Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x),
                Lerp(rng.Uniform<float>(), Range.pMin.x, Range.pMax.x)),
                Dir);
            InvDirs[i].InvDir = Vector3f(1.0f / Dir.x, 1.0f / Dir.y, 1.0f / Dir.z);

            InvDirs[i].DirIsNeg[0] = Dir.x > 0 ? 1 : 0;
            InvDirs[i].DirIsNeg[1] = Dir.y > 0 ? 1 : 0;
            InvDirs[i].DirIsNeg[2] = Dir.z > 0 ? 1 : 0;
        }
    }
};

const float TestWorldSize = 65535.0f;
const float HalfTestWorldSize = TestWorldSize * 0.5f;

const FRandomBounds GRandomBounds(
    8 << 10, Bounds3f(Point3f(-TestWorldSize, -TestWorldSize, -TestWorldSize),
                   Point3f(+TestWorldSize, +TestWorldSize, +TestWorldSize)));
const FRandomTriangle GRandomTriangle(
    8 << 10, Bounds3f(Point3f(-TestWorldSize, -TestWorldSize, -TestWorldSize),
                   Point3f(+TestWorldSize, +TestWorldSize, +TestWorldSize)));
const FRandomRays GRandomRays(
    1024, Bounds3f(Point3f(-HalfTestWorldSize, -HalfTestWorldSize, -HalfTestWorldSize),
                   Point3f(+HalfTestWorldSize, +HalfTestWorldSize, +HalfTestWorldSize)));

static int64_t Intersections = 0;
static const float GINFINITY = std::numeric_limits<float>::infinity();

static void BM_RandomBoundsRayIntersection(benchmark::State& state) {
    int64_t BoundsCount = state.range(0);
    int64_t RayCount = state.range(1);

    for (auto _ : state) {
        for (int64_t RayIndex = 0; RayIndex < RayCount; ++RayIndex) {
            const Ray& Ray = GRandomRays.Rays[RayIndex];

            for (int64_t BoundsIndex = 0; BoundsIndex < BoundsCount; ++BoundsIndex) {
                const Bounds3f& Bounds3 = GRandomBounds.Bounds[BoundsIndex];
                Intersections += Bounds3.IntersectP(Ray.o, Ray.d, GINFINITY) ? 1 : 0;
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * BoundsCount * RayCount);
}
BENCHMARK(BM_RandomBoundsRayIntersection)->Ranges({{1 << 10, 8 << 10}, {256, 1024}});

static void BM_RandomBoundsRayIntersectionTest(benchmark::State& state) {
    int64_t BoundsCount = state.range(0);
    int64_t RayCount = state.range(1);

    for (auto _ : state) {
        for (int64_t RayIndex = 0; RayIndex < RayCount; ++RayIndex) {
            const Ray& Ray = GRandomRays.Rays[RayIndex];
            const auto& InvDir = GRandomRays.InvDirs[RayIndex];

            for (int64_t BoundsIndex = 0; BoundsIndex < BoundsCount; ++BoundsIndex) {
                const Bounds3f& Bounds3 = GRandomBounds.Bounds[BoundsIndex];
                Intersections += Bounds3.IntersectP(Ray.o, Ray.d, GINFINITY,
                                                    InvDir.InvDir, InvDir.DirIsNeg)
                                     ? 1
                                     : 0;
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * BoundsCount * RayCount);
}
BENCHMARK(BM_RandomBoundsRayIntersectionTest)->Ranges({{1 << 10, 8 << 10}, {256, 1024}});

static void BM_RandomTriangleRayIntersection(benchmark::State& state) {
    int64_t TriangleCount = state.range(0);
    int64_t RayCount = state.range(1);

    for (auto _ : state) {
        for (int64_t RayIndex = 0; RayIndex < RayCount; ++RayIndex) {
            const Ray& Ray = GRandomRays.Rays[RayIndex];

            for (int64_t BoundsIndex = 0; BoundsIndex < TriangleCount; ++BoundsIndex) {
                const auto& Shape = GRandomTriangle.Shapes[BoundsIndex];

                float Hit = GINFINITY;
                Intersections += Shape.Intersect(Ray, Hit).has_value() ? 1 : 0;
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * TriangleCount * RayCount);
}
BENCHMARK(BM_RandomTriangleRayIntersection)->Ranges({{1 << 10, 8 << 10}, {256, 1024}});

static void BM_RandomTriangleRayIntersectionTest(benchmark::State& state) {
    int64_t TriangleCount = state.range(0);
    int64_t RayCount = state.range(1);

    for (auto _ : state) {
        for (int64_t RayIndex = 0; RayIndex < RayCount; ++RayIndex) {
            const Ray& Ray = GRandomRays.Rays[RayIndex];

            for (int64_t BoundsIndex = 0; BoundsIndex < TriangleCount; ++BoundsIndex) {
                const auto& Shape = GRandomTriangle.Shapes[BoundsIndex];

                Intersections += Shape.IntersectP(Ray) ? 1 : 0;
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * TriangleCount * RayCount);
}
BENCHMARK(BM_RandomTriangleRayIntersectionTest)->Ranges({{1 << 10, 8 << 10}, {256, 1024}});
