#ifndef DY_SOLVER_LDL_H
#define DY_SOLVER_LDL_H

#include "DySolverCore.h"
#include "DySolverConstraintDesc.h"

#include "foundation/PxPreprocessor.h"

//#include "LogSystem.h"

namespace physx
{
	namespace Dy
	{
#define DYNAMIC_ARTICULATION_REGISTRATIONLDL(x) 0

		struct JointNode {
			PxU32 jointIndex = 0;          // pointer to enclosing dxJoint object
			PxU32 otherBodyIndex = 0;      // *other* body this joint is connected to
			PxI32 nextJointNodeIndex = -1;  //-1 for null
			PxU32 constraintRowIndex = 0;
		};

		typedef void (*SolveBlockMethodLDL)(const SolverIslandParams& params, const PxSolverConstraintDesc* desc, const PxU32 constraintCount, SolverContext& cache);

		class SolverCoreLDL : public SolverCore
		{
		public:
			bool frictionEveryIteration;
			static SolverCoreLDL* create(bool fricEveryIteration);

			// Implements SolverCore
			virtual void destroyV();

			virtual PxI32 solveVParallelAndWriteBack
			(SolverIslandParams& params, Cm::SpatialVectorF* Z, Cm::SpatialVectorF* deltaV) const;

			virtual void solveV_Blocks
			(SolverIslandParams& params) const;

			virtual void writeBackV
			(const PxSolverConstraintDesc* PX_RESTRICT constraintList, const PxU32 constraintListSize, PxConstraintBatchHeader* contactConstraintBatches, const PxU32 numBatches,
				ThresholdStreamElement* PX_RESTRICT thresholdStream, const PxU32 thresholdStreamLength, PxU32& outThresholdPairs,
				PxSolverBodyData* atomListData, WriteBackBlockMethod writeBackTable[]) const;

		private:

			//~Implements SolverCore
		};
	}
}

#endif
