#ifndef APO_GPU_ALGORITHM_CUH
#define APO_GPU_ALGORITHM_CUH

#include "apo/core/math.hpp"
#include "apo/core/type.hpp"
#include "apo/gpu/algorithm.hpp"
#include "apo/gpu/lbvh.cuh"
#include "apo/gpu/memory.cuh"

namespace apo::gpu
{
    template<uint32_t K = 16>
    struct AlgorithmGPU : public Algorithm
    {
      public:
        AlgorithmGPU( ConstSpan<Real> sites, uint32_t knnValidationStep = 1 );

        struct Sample
        {
            double initialization;
            double bvhConstruction;
            double knnSearch;
            double knnConstruction;
            double vertexValidation;
            double edgeValidation;
            double bisectorValidation;
        };

        virtual void          build() override;
        virtual VertexDiagram toHost( bool doValidation ) override;
        virtual FullDiagram   toHostFull( bool doValidation ) override;
        virtual Topology      toTopology() override;

        uint32_t knnValidationStep;
        LBVH     bvh;

        DeviceBuffer dSites;
        DeviceBuffer dStatus;
        DeviceBuffer dVertices;
        DeviceBuffer dVerticesNb;
        DeviceBuffer dEdges;
        DeviceBuffer dEdgesNb;

        // For benchmarking purpose
        Sample sample;
    };
} // namespace apo::gpu

#include "apo/gpu/algorithm.inl"

#endif // APO_GPU_ALGORITHM_CUH