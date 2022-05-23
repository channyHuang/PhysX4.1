#ifndef DY_DYNAMICS_LDL_PRE_H
#define DY_DYNAMICS_LDL_PRE_H

#include "DyDynamicsLDL.h"

#include "PsTime.h"
#include "PsAtomic.h"
#include "PxvDynamics.h"

#include "common/PxProfileZone.h"
#include "PxsRigidBody.h"
#include "PxsContactManager.h"
#include "DyDynamics.h"
#include "DyBodyCoreIntegrator.h"
#include "DySolverCore.h"
#include "DySolverControl.h"
#include "DySolverContact.h"
#include "DySolverContactPF.h"
#include "DyArticulationContactPrep.h"
#include "DySolverBody.h"

#include "DyConstraintPrep.h"
#include "DyConstraintPartition.h"
#include "DyArticulation.h"

#include "CmFlushPool.h"
#include "DyArticulationPImpl.h"
#include "PxsMaterialManager.h"
#include "DySolverContactPF4.h"
#include "DyContactReduction.h"
#include "PxcNpContactPrepShared.h"
#include "DyContactPrep.h"
#include "DySolverControlPF.h"
#include "PxSceneDesc.h"
#include "PxsSimpleIslandManager.h"
#include "PxvNphaseImplementationContext.h"
#include "PxvSimStats.h"
#include "PxsContactManagerState.h"
#include "PxsDefaultMemoryManager.h"
#include "DyContactPrepShared.h"

namespace physx
{
	namespace Dy {
		struct SolverIslandObjects
		{
			PxsRigidBody** bodies;
			ArticulationV** articulations;
			Dy::ArticulationV** articulationOwners;
			PxsIndexedContactManager* contactManagers;
			//PxsIndexedConstraint*		constraints;

			const IG::IslandId* islandIds;
			PxU32						numIslands;
			PxU32* bodyRemapTable;
			PxU32* nodeIndexArray;

			PxSolverConstraintDesc* constraintDescs;
			PxSolverConstraintDesc* orderedConstraintDescs;
			PxSolverConstraintDesc* tempConstraintDescs;
			PxConstraintBatchHeader* constraintBatchHeaders;
			Cm::SpatialVector* motionVelocities;
			PxsBodyCore** bodyCoreArray;

			SolverIslandObjects() : bodies(NULL), articulations(NULL), articulationOwners(NULL),
				contactManagers(NULL), islandIds(NULL), numIslands(0), nodeIndexArray(NULL), constraintDescs(NULL), orderedConstraintDescs(NULL),
				tempConstraintDescs(NULL), constraintBatchHeaders(NULL), motionVelocities(NULL), bodyCoreArray(NULL)
			{
			}
		};
		struct ArticulationSortPredicate
		{
			bool operator()(const PxsIndexedContactManager*& left, const PxsIndexedContactManager*& right) const
			{
				return left->contactManager->getWorkUnit().index < right->contactManager->getWorkUnit().index;
			}
		};
		struct ConstraintLess
		{
			bool operator()(const PxSolverConstraintDesc& left, const PxSolverConstraintDesc& right) const
			{
				return reinterpret_cast<Constraint*>(left.constraint)->index > reinterpret_cast<Constraint*>(right.constraint)->index;
			}
		};
		struct EnhancedSortPredicate
		{
			bool operator()(const PxsIndexedContactManager& left, const PxsIndexedContactManager& right) const
			{
				PxcNpWorkUnit& unit0 = left.contactManager->getWorkUnit();
				PxcNpWorkUnit& unit1 = right.contactManager->getWorkUnit();
				return (unit0.mTransformCache0 < unit1.mTransformCache0) ||
					((unit0.mTransformCache0 == unit1.mTransformCache0) && (unit0.mTransformCache1 < unit1.mTransformCache1));
			}
		};
		
		class PxsPreIntegrateTask : public Cm::Task
		{
			PxsPreIntegrateTask& operator=(const PxsPreIntegrateTask&);
		public:
			PxsPreIntegrateTask(DynamicsLDLContext& context,
				PxsBodyCore* const* bodyArray,
				PxsRigidBody* const* originalBodyArray,
				PxU32 const* nodeIndexArray,
				PxSolverBody* solverBodies,
				PxSolverBodyData* solverBodyDataPool,
				PxF32				dt,
				PxU32				numBodies,
				volatile PxU32* maxSolverPositionIterations,
				volatile PxU32* maxSolverVelocityIterations,
				const PxU32			startIndex,
				const PxU32			numToIntegrate,
				const PxVec3& gravity) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mBodyArray(bodyArray),
				mOriginalBodyArray(originalBodyArray),
				mNodeIndexArray(nodeIndexArray),
				mSolverBodies(solverBodies),
				mSolverBodyDataPool(solverBodyDataPool),
				mDt(dt),
				mNumBodies(numBodies),
				mMaxSolverPositionIterations(maxSolverPositionIterations),
				mMaxSolverVelocityIterations(maxSolverVelocityIterations),
				mStartIndex(startIndex),
				mNumToIntegrate(numToIntegrate),
				mGravity(gravity)
			{}

			virtual void runInternal();

			virtual const char* getName() const
			{
				return "PxsDynamics.preIntegrate";
			}

		public:
			DynamicsLDLContext& mContext;
			PxsBodyCore* const* mBodyArray;
			PxsRigidBody* const* mOriginalBodyArray;
			PxU32 const* mNodeIndexArray;
			PxSolverBody* mSolverBodies;
			PxSolverBodyData* mSolverBodyDataPool;
			PxF32					mDt;
			PxU32					mNumBodies;
			volatile PxU32* mMaxSolverPositionIterations;
			volatile PxU32* mMaxSolverVelocityIterations;
			PxU32					mStartIndex;
			PxU32					mNumToIntegrate;
			PxVec3					mGravity;
		};
		void solveParallel(SOLVER_PARALLEL_METHOD_ARGS_LDL)
		{
			Dy::ThreadContext& threadContext = *context.getThreadContext();
			threadContext.mZVector.forceSize_Unsafe(0);
			threadContext.mZVector.reserve(params.mMaxArticulationLinks);
			threadContext.mZVector.forceSize_Unsafe(params.mMaxArticulationLinks);

			threadContext.mDeltaV.forceSize_Unsafe(0);
			threadContext.mDeltaV.reserve(params.mMaxArticulationLinks);
			threadContext.mDeltaV.forceSize_Unsafe(params.mMaxArticulationLinks);

			context.solveParallel(params, islandSim, threadContext.mZVector.begin(), threadContext.mDeltaV.begin());

			context.putThreadContext(&threadContext);
		}

		class PxsParallelSolverTask : public Cm::Task
		{
			PxsParallelSolverTask& operator=(PxsParallelSolverTask&);
		public:

			PxsParallelSolverTask(SolverIslandParams& params, DynamicsLDLContext& context, PxFrictionType::Enum frictionType, IG::IslandSim& islandSim)
				: Cm::Task(context.getContextId()), mParams(params), mContext(context), mFrictionType(frictionType), mIslandSim(islandSim)
			{
			}

			virtual void runInternal()
			{
				solveParallel(mContext, mParams, mIslandSim);
			}

			virtual const char* getName() const
			{
				return "PxsDynamics.parallelSolver";
			}

			SolverIslandParams& mParams;
			DynamicsLDLContext& mContext;
			PxFrictionType::Enum	mFrictionType;
			IG::IslandSim& mIslandSim;
		};

		class SolverArticulationUpdateTask : public Cm::Task
		{
			ThreadContext& mIslandThreadContext;

			ArticulationV** mArticulations;
			ArticulationSolverDesc* mArticulationDescArray;
			PxU32 mNbToProcess;

			Dy::DynamicsLDLContext& mContext;
			PxU32 mStartIdx;

		public:

			static const PxU32 NbArticulationsPerTask = 32;

			SolverArticulationUpdateTask(ThreadContext& islandThreadContext, ArticulationV** articulations, ArticulationSolverDesc* articulationDescArray, PxU32 nbToProcess, Dy::DynamicsLDLContext& context,
				PxU32 startIdx) :
				Cm::Task(context.getContextId()), mIslandThreadContext(islandThreadContext), mArticulations(articulations), mArticulationDescArray(articulationDescArray), mNbToProcess(nbToProcess), mContext(context), mStartIdx(startIdx)
			{
			}

			virtual const char* getName() const { return "SolverArticulationUpdateTask"; }

			virtual void runInternal()
			{
				ThreadContext& threadContext = *mContext.getThreadContext();

				threadContext.mConstraintBlockStream.reset(); //Clear in case there's some left-over memory in this context, for which the block has already been freed 
				PxU32 maxVelIters = 0;
				PxU32 maxPosIters = 0;
				PxU32 maxArticulationLength = 0;
				PxU32 maxSolverArticLength = 0;
				PxU32 maxLinks = 0;

				for (PxU32 i = 0; i < mNbToProcess; i++)
				{
					ArticulationV& a = *(mArticulations[i]);
					a.getSolverDesc(mArticulationDescArray[i]);

					maxLinks = PxMax(maxLinks, PxU32(mArticulationDescArray[i].linkCount));
				}

				threadContext.mZVector.forceSize_Unsafe(0);
				threadContext.mZVector.reserve(maxLinks);
				threadContext.mZVector.forceSize_Unsafe(maxLinks);

				threadContext.mDeltaV.forceSize_Unsafe(0);
				threadContext.mDeltaV.reserve(maxLinks);
				threadContext.mDeltaV.forceSize_Unsafe(maxLinks);

				PxU32 startIdx = mStartIdx;

				BlockAllocator blockAllocator(mIslandThreadContext.mConstraintBlockManager, threadContext.mConstraintBlockStream, threadContext.mFrictionPatchStreamPair, threadContext.mConstraintSize);

				for (PxU32 i = 0; i < mNbToProcess; i++)
				{
					ArticulationV& a = *(mArticulations[i]);

					PxU32 acCount, descCount;

					descCount = ArticulationPImpl::computeUnconstrainedVelocities(mArticulationDescArray[i], mContext.mDt, blockAllocator,
						mIslandThreadContext.mContactDescPtr + startIdx, acCount, mContext.getScratchAllocator(),
						mContext.getGravity(), mContext.getContextId(),
						threadContext.mZVector.begin(), threadContext.mDeltaV.begin());

					mArticulationDescArray[i].numInternalConstraints = Ps::to8(descCount);

					maxArticulationLength = PxMax(maxArticulationLength, PxU32(mArticulationDescArray[i].totalDataSize));
					maxSolverArticLength = PxMax(maxSolverArticLength, PxU32(mArticulationDescArray[i].solverDataSize));
					//maxLinks = PxMax(maxLinks, PxU32(mArticulationDescArray[i].linkCount));

					const PxU16 iterWord = a.getIterationCounts();
					maxVelIters = PxMax<PxU32>(PxU32(iterWord >> 8), maxVelIters);
					maxPosIters = PxMax<PxU32>(PxU32(iterWord & 0xff), maxPosIters);
					startIdx += DY_ARTICULATION_MAX_SIZE;
				}

				Ps::atomicMax(reinterpret_cast<PxI32*>(&mIslandThreadContext.mMaxSolverPositionIterations), PxI32(maxPosIters));
				Ps::atomicMax(reinterpret_cast<PxI32*>(&mIslandThreadContext.mMaxSolverVelocityIterations), PxI32(maxVelIters));
				Ps::atomicMax(reinterpret_cast<PxI32*>(&mIslandThreadContext.mMaxArticulationLength), PxI32(maxArticulationLength));
				Ps::atomicMax(reinterpret_cast<PxI32*>(&mIslandThreadContext.mMaxArticulationSolverLength), PxI32(maxSolverArticLength));
				Ps::atomicMax(reinterpret_cast<PxI32*>(&mIslandThreadContext.mMaxArticulationLinks), PxI32(maxLinks));

				mContext.putThreadContext(&threadContext);
			}

		private:
			PX_NOCOPY(SolverArticulationUpdateTask)
		};
		
		class PxsSolverCreateFinalizeConstraintsTask : public Cm::Task
		{
			PxsSolverCreateFinalizeConstraintsTask& operator=(const PxsSolverCreateFinalizeConstraintsTask&);
		public:

			PxsSolverCreateFinalizeConstraintsTask(
				DynamicsLDLContext& context,
				IslandContext& islandContext,
				PxU32 solverDataOffset,
				PxsContactManagerOutputIterator& outputs,
				bool enhancedDeterminism) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mIslandContext(islandContext),
				mSolverDataOffset(solverDataOffset),
				mOutputs(outputs),
				mEnhancedDeterminism(enhancedDeterminism)
			{
			}

			virtual void runInternal();

			virtual const char* getName() const { return "PxsDynamics.solverCreateFinalizeConstraints"; }

			DynamicsLDLContext& mContext;
			IslandContext& mIslandContext;
			PxU32								mSolverDataOffset;
			PxsContactManagerOutputIterator& mOutputs;
			bool								mEnhancedDeterminism;
		};

		class PxsSolverSetupSolveTask : public Cm::Task
		{
			PxsSolverSetupSolveTask& operator=(const PxsSolverSetupSolveTask&);
		public:

			PxsSolverSetupSolveTask(
				DynamicsLDLContext& context,
				IslandContext& islandContext,
				const SolverIslandObjects& objects,
				const PxU32 solverBodyOffset,
				IG::IslandSim& islandSim) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mIslandContext(islandContext),
				mObjects(objects),
				mSolverBodyOffset(solverBodyOffset),
				mIslandSim(islandSim)
			{}

			virtual void runInternal()
			{
				ThreadContext& mThreadContext = *mIslandContext.mThreadContext;

				PxSolverConstraintDesc* contactDescBegin = mThreadContext.orderedContactConstraints;
				PxSolverConstraintDesc* contactDescPtr = mThreadContext.orderedContactConstraints;

				PxSolverBody* solverBodies = mContext.mSolverBodyPool.begin() + mSolverBodyOffset;
				PxSolverBodyData* solverBodyDatas = mContext.mSolverBodyDataPool.begin();

				PxU32 frictionDescCount = mThreadContext.mNumDifferentBodyFrictionConstraints;

				PxU32 j = 0, i = 0;

				//On PS3, self-constraints will be bumped to the end of the constraint list
				//and processed separately. On PC/360, they will be mixed in the array and
				//classed as "different body" constraints regardless of the fact that they're self-constraints.
				//PxU32 numBatches = mThreadContext.numDifferentBodyBatchHeaders;
				// TODO: maybe replace with non-null joints from end of the array

				PxU32 numBatches = 0;

				PxU32 currIndex = 0;
				for (PxU32 a = 0; a < mThreadContext.mConstraintsPerPartition.size(); ++a)
				{
					PxU32 endIndex = currIndex + mThreadContext.mConstraintsPerPartition[a];

					PxU32 numBatchesInPartition = 0;
					for (PxU32 b = currIndex; b < endIndex; ++b)
					{
						PxConstraintBatchHeader& _header = mThreadContext.contactConstraintBatchHeaders[b];
						PxU16 stride = _header.stride, newStride = _header.stride;
						PxU32 startIndex = j;
						for (PxU16 c = 0; c < stride; ++c)
						{
							if (getConstraintLength(contactDescBegin[i]) == 0)
							{
								newStride--;
								i++;
							}
							else
							{
								if (i != j)
									contactDescBegin[j] = contactDescBegin[i];
								i++;
								j++;
								contactDescPtr++;
							}
						}

						if (newStride != 0)
						{
							mThreadContext.contactConstraintBatchHeaders[numBatches].startIndex = startIndex;
							mThreadContext.contactConstraintBatchHeaders[numBatches].stride = newStride;
							PxU8 type = *contactDescBegin[startIndex].constraint;
							if (type == DY_SC_TYPE_STATIC_CONTACT)
							{
								//Check if any block of constraints is classified as type static (single) contact constraint.
								//If they are, iterate over all constraints grouped with it and switch to "dynamic" contact constraint
								//type if there's a dynamic contact constraint in the group.
								for (PxU32 c = 1; c < newStride; ++c)
								{
									if (*contactDescBegin[startIndex + c].constraint == DY_SC_TYPE_RB_CONTACT)
									{
										type = DY_SC_TYPE_RB_CONTACT;
									}
								}
							}

							mThreadContext.contactConstraintBatchHeaders[numBatches].constraintType = type;
							numBatches++;
							numBatchesInPartition++;
						}
					}
					PxU32 numHeaders = numBatchesInPartition;
					currIndex += mThreadContext.mConstraintsPerPartition[a];
					mThreadContext.mConstraintsPerPartition[a] = numHeaders;
				}

				PxU32 contactDescCount = PxU32(contactDescPtr - contactDescBegin);

				mThreadContext.mNumDifferentBodyConstraints = contactDescCount;

				mThreadContext.numContactConstraintBatches = numBatches;
				mThreadContext.mNumSelfConstraints = j - contactDescCount; //self constraint count
				contactDescCount = j;
				mThreadContext.mOrderedContactDescCount = j;

				//Now do the friction constraints if we're not using the sticky model
				if (mContext.getFrictionType() != PxFrictionType::ePATCH)
				{
					PxSolverConstraintDesc* frictionDescBegin = mThreadContext.frictionConstraintDescArray.begin();
					PxSolverConstraintDesc* frictionDescPtr = frictionDescBegin;

					Ps::Array<PxConstraintBatchHeader>& frictionHeaderArray = mThreadContext.frictionConstraintBatchHeaders;
					frictionHeaderArray.forceSize_Unsafe(0);
					frictionHeaderArray.reserve(mThreadContext.numContactConstraintBatches);
					PxConstraintBatchHeader* headers = frictionHeaderArray.begin();

					Ps::Array<PxU32>& constraintsPerPartition = mThreadContext.mConstraintsPerPartition;
					Ps::Array<PxU32>& frictionConstraintsPerPartition = mThreadContext.mFrictionConstraintsPerPartition;
					frictionConstraintsPerPartition.forceSize_Unsafe(0);
					frictionConstraintsPerPartition.reserve(constraintsPerPartition.capacity());

					PxU32 fricI = 0;
					PxU32 startIndex = 0;
					PxU32 fricHeaders = 0;
					for (PxU32 k = 0; k < constraintsPerPartition.size(); ++k)
					{
						PxU32 numBatchesInK = constraintsPerPartition[k];
						PxU32 endIndex = startIndex + numBatchesInK;

						PxU32 startFricH = fricHeaders;

						for (PxU32 a = startIndex; a < endIndex; ++a)
						{
							PxConstraintBatchHeader& _header = mThreadContext.contactConstraintBatchHeaders[a];
							PxU16 stride = _header.stride;
							if (_header.constraintType == DY_SC_TYPE_RB_CONTACT || _header.constraintType == DY_SC_TYPE_EXT_CONTACT ||
								_header.constraintType == DY_SC_TYPE_STATIC_CONTACT)
							{
								PxU8 type = 0;
								//Extract friction from this constraint
								for (PxU16 b = 0; b < stride; ++b)
								{
									//create the headers...
									PxSolverConstraintDesc& desc = contactDescBegin[_header.startIndex + b];
									PX_ASSERT(desc.constraint);
									SolverContactCoulombHeader* header = reinterpret_cast<SolverContactCoulombHeader*>(desc.constraint);
									PxU32 frictionOffset = header->frictionOffset;
									PxU8* PX_RESTRICT constraint = reinterpret_cast<PxU8*>(header) + frictionOffset;
									const PxU32 origLength = getConstraintLength(desc);
									const PxU32 length = (origLength - frictionOffset);

									setConstraintLength(*frictionDescPtr, length);
									frictionDescPtr->constraint = constraint;
									frictionDescPtr->bodyA = desc.bodyA;
									frictionDescPtr->bodyB = desc.bodyB;
									frictionDescPtr->bodyADataIndex = desc.bodyADataIndex;
									frictionDescPtr->bodyBDataIndex = desc.bodyBDataIndex;
									frictionDescPtr->linkIndexA = desc.linkIndexA;
									frictionDescPtr->linkIndexB = desc.linkIndexB;
									frictionDescPtr->writeBack = NULL;
									frictionDescPtr->writeBackLengthOver4 = 0;
									type = *constraint;
									frictionDescPtr++;
								}
								headers->startIndex = fricI;
								headers->stride = stride;
								headers->constraintType = type;
								headers++;
								fricHeaders++;
								fricI += stride;
							}
							else if (_header.constraintType == DY_SC_TYPE_BLOCK_RB_CONTACT || _header.constraintType == DY_SC_TYPE_BLOCK_STATIC_RB_CONTACT)
							{
								//KS - TODO - Extract block of 4 contacts from this constraint. This isn't implemented yet for coulomb friction model
								PX_ASSERT(contactDescBegin[_header.startIndex].constraint);
								SolverContactCoulombHeader4* head = reinterpret_cast<SolverContactCoulombHeader4*>(contactDescBegin[_header.startIndex].constraint);
								PxU32 frictionOffset = head->frictionOffset;
								PxU8* PX_RESTRICT constraint = reinterpret_cast<PxU8*>(head) + frictionOffset;
								const PxU32 origLength = getConstraintLength(contactDescBegin[_header.startIndex]);
								const PxU32 length = (origLength - frictionOffset);
								PxU8 type = *constraint;
								PX_ASSERT(type == DY_SC_TYPE_BLOCK_FRICTION || type == DY_SC_TYPE_BLOCK_STATIC_FRICTION);
								for (PxU32 b = 0; b < 4; ++b)
								{
									PxSolverConstraintDesc& desc = contactDescBegin[_header.startIndex + b];
									setConstraintLength(*frictionDescPtr, length);
									frictionDescPtr->constraint = constraint;
									frictionDescPtr->bodyA = desc.bodyA;
									frictionDescPtr->bodyB = desc.bodyB;
									frictionDescPtr->bodyADataIndex = desc.bodyADataIndex;
									frictionDescPtr->bodyBDataIndex = desc.bodyBDataIndex;
									frictionDescPtr->linkIndexA = desc.linkIndexA;
									frictionDescPtr->linkIndexB = desc.linkIndexB;
									frictionDescPtr->writeBack = NULL;
									frictionDescPtr->writeBackLengthOver4 = 0;
									frictionDescPtr++;
								}
								headers->startIndex = fricI;
								headers->stride = stride;
								headers->constraintType = type;
								headers++;
								fricHeaders++;
								fricI += stride;
							}
						}
						startIndex += numBatchesInK;
						if (startFricH < fricHeaders)
						{
							frictionConstraintsPerPartition.pushBack(fricHeaders - startFricH);
						}
					}

					frictionDescCount = PxU32(frictionDescPtr - frictionDescBegin);

					mThreadContext.mNumDifferentBodyFrictionConstraints = frictionDescCount;

					frictionHeaderArray.forceSize_Unsafe(PxU32(headers - frictionHeaderArray.begin()));

					mThreadContext.mNumSelfFrictionConstraints = fricI - frictionDescCount; //self constraint count
					mThreadContext.mNumDifferentBodyFrictionConstraints = frictionDescCount;
					frictionDescCount = fricI;
					mThreadContext.mOrderedFrictionDescCount = frictionDescCount;
				}

				{
					{
						PX_PROFILE_ZONE("Dynamics.solver", mContext.getContextId());

						PxSolverConstraintDesc* contactDescs = mThreadContext.orderedContactConstraints;
						PxSolverConstraintDesc* frictionDescs = mThreadContext.frictionConstraintDescArray.begin();

						PxI32* thresholdPairsOut = &mContext.mThresholdStreamOut;

						SolverIslandParams& params = *reinterpret_cast<SolverIslandParams*>(mContext.getTaskPool().allocate(sizeof(SolverIslandParams)));
						params.positionIterations = mThreadContext.mMaxSolverPositionIterations;
						params.velocityIterations = mThreadContext.mMaxSolverVelocityIterations;
						params.bodyListStart = solverBodies;
						params.bodyDataList = solverBodyDatas;
						params.solverBodyOffset = mSolverBodyOffset;
						params.bodyListSize = mIslandContext.mCounts.bodies;
						params.articulationListStart = mThreadContext.getArticulations().begin();
						params.articulationListSize = mThreadContext.getArticulations().size();
						params.constraintList = contactDescs;
						params.constraintIndex = 0;
						params.constraintIndex2 = 0;
						params.bodyListIndex = 0;
						params.bodyListIndex2 = 0;
						params.articSolveIndex = 0;
						params.articSolveIndex2 = 0;
						params.bodyIntegrationListIndex = 0;
						params.thresholdStream = mContext.getThresholdStream().begin();
						params.thresholdStreamLength = mContext.getThresholdStream().size();
						params.outThresholdPairs = thresholdPairsOut;
						params.motionVelocityArray = mThreadContext.motionVelocityArray;
						params.bodyArray = mThreadContext.mBodyCoreArray;
						params.numObjectsIntegrated = 0;
						params.constraintBatchHeaders = mThreadContext.contactConstraintBatchHeaders;
						params.numConstraintHeaders = mThreadContext.numContactConstraintBatches;
						params.headersPerPartition = mThreadContext.mConstraintsPerPartition.begin();
						params.nbPartitions = mThreadContext.mConstraintsPerPartition.size();
						params.rigidBodies = const_cast<PxsRigidBody**>(mObjects.bodies);
						params.frictionHeadersPerPartition = mThreadContext.mFrictionConstraintsPerPartition.begin();
						params.nbFrictionPartitions = mThreadContext.mFrictionConstraintsPerPartition.size();
						params.frictionConstraintBatches = mThreadContext.frictionConstraintBatchHeaders.begin();
						params.numFrictionConstraintHeaders = mThreadContext.frictionConstraintBatchHeaders.size();
						params.frictionConstraintIndex = 0;
						params.frictionConstraintList = frictionDescs;
						params.mMaxArticulationLinks = mThreadContext.mMaxArticulationLinks;
						params.dt = mContext.mDt;
						params.invDt = mContext.mInvDt;

						const PxU32 unrollSize = 8;
						const PxU32 denom = PxMax(1u, (mThreadContext.mMaxPartitions * unrollSize));
						const PxU32 MaxTasks = getTaskManager()->getCpuDispatcher()->getWorkerCount();
						const PxU32 idealThreads = (mThreadContext.numContactConstraintBatches + denom - 1) / denom;
						const PxU32 numTasks = PxMax(1u, PxMin(idealThreads, MaxTasks));

						if (numTasks > 1)
						{
							const PxU32 idealBatchSize = PxMax(unrollSize, idealThreads * unrollSize / (numTasks * 2));

							params.batchSize = idealBatchSize; //assigning ideal batch size for the solver to grab work at. Only needed by the multi-threaded island solver.

							for (PxU32 a = 1; a < numTasks; ++a)
							{
								void* tsk = mContext.getTaskPool().allocate(sizeof(PxsParallelSolverTask));
								PxsParallelSolverTask* pTask = PX_PLACEMENT_NEW(tsk, PxsParallelSolverTask)(
									params, mContext, mContext.getFrictionType(), mIslandSim);

								//Force to complete before merge task!
								pTask->setContinuation(mCont);

								pTask->removeReference();
							}

							//Avoid kicking off one parallel task when we can do the work inline in this function
							{
								PX_PROFILE_ZONE("Dynamics.parallelSolve", mContext.getContextId());

								solveParallel(mContext, params, mIslandSim);
							}
							const PxI32 numBodiesPlusArtics = PxI32(mIslandContext.mCounts.bodies + mIslandContext.mCounts.articulations);

							PxI32* numObjectsIntegrated = &params.numObjectsIntegrated;

							WAIT_FOR_PROGRESS_NO_TIMER(numObjectsIntegrated, numBodiesPlusArtics);
						}
						else
						{
							mThreadContext.mZVector.forceSize_Unsafe(0);
							mThreadContext.mZVector.reserve(mThreadContext.mMaxArticulationLinks);
							mThreadContext.mZVector.forceSize_Unsafe(mThreadContext.mMaxArticulationLinks);

							mThreadContext.mDeltaV.forceSize_Unsafe(0);
							mThreadContext.mDeltaV.reserve(mThreadContext.mMaxArticulationLinks);
							mThreadContext.mDeltaV.forceSize_Unsafe(mThreadContext.mMaxArticulationLinks);

							params.Z = mThreadContext.mZVector.begin();
							params.deltaV = mThreadContext.mDeltaV.begin();

							//Only one task - a small island so do a sequential solve (avoid the atomic overheads)
							solveVBlock(mContext.mSolverCore[mContext.getFrictionType()], params);

							const PxU32 bodyCountMin1 = mIslandContext.mCounts.bodies - 1u;
							PxSolverBodyData* solverBodyData2 = solverBodyDatas + mSolverBodyOffset + 1;
							for (PxU32 k = 0; k < mIslandContext.mCounts.bodies; k++)
							{
								const PxU32 prefetchAddress = PxMin(k + 4, bodyCountMin1);
								Ps::prefetchLine(mThreadContext.mBodyCoreArray[prefetchAddress]);
								Ps::prefetchLine(&mThreadContext.motionVelocityArray[k], 128);
								Ps::prefetchLine(&mThreadContext.mBodyCoreArray[prefetchAddress], 128);
								Ps::prefetchLine(&mObjects.bodies[prefetchAddress]);

								PxSolverBodyData& solverBodyData = solverBodyData2[k];

								integrateCore(mThreadContext.motionVelocityArray[k].linear, mThreadContext.motionVelocityArray[k].angular,
									solverBodies[k], solverBodyData, mContext.mDt);

								PxsRigidBody& rBody = *mObjects.bodies[k];
								PxsBodyCore& core = rBody.getCore();
								rBody.mLastTransform = core.body2World;
								core.body2World = solverBodyData.body2World;
								core.linearVelocity = solverBodyData.linearVelocity;
								core.angularVelocity = solverBodyData.angularVelocity;

								bool hasStaticTouch = mIslandSim.getIslandStaticTouchCount(IG::NodeIndex(solverBodyData.nodeIndex)) != 0;
								sleepCheck(const_cast<PxsRigidBody*>(mObjects.bodies[k]), mContext.mDt, mContext.mInvDt, mContext.mEnableStabilization, mContext.mUseAdaptiveForce, mThreadContext.motionVelocityArray[k],
									hasStaticTouch);
							}

							for (PxU32 cnt = 0; cnt < mIslandContext.mCounts.articulations; cnt++)
							{
								ArticulationSolverDesc& d = mThreadContext.getArticulations()[cnt];
								//PX_PROFILE_ZONE("Articulations.integrate", mContext.getContextId());

								ArticulationPImpl::updateBodies(d, mContext.getDt());
							}
						}
					}
				}
			}

			virtual const char* getName() const { return "PxsDynamics.solverSetupSolve"; }

			DynamicsLDLContext& mContext;
			IslandContext& mIslandContext;
			const SolverIslandObjects	mObjects;
			PxU32						mSolverBodyOffset;
			IG::IslandSim& mIslandSim;
		};
		
		class PxsSolverConstraintPartitionTask : public Cm::Task
		{
			PxsSolverConstraintPartitionTask& operator=(const PxsSolverConstraintPartitionTask&);
		public:

			PxsSolverConstraintPartitionTask(DynamicsLDLContext& context,
				IslandContext& islandContext,
				const SolverIslandObjects& objects,
				const PxU32 solverBodyOffset, bool enhancedDeterminism) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mIslandContext(islandContext),
				mObjects(objects),
				mSolverBodyOffset(solverBodyOffset),
				mEnhancedDeterminism(enhancedDeterminism)
			{}

			virtual void runInternal()
			{
				PX_PROFILE_ZONE("PartitionConstraints", mContext.getContextId());
				ThreadContext& mThreadContext = *mIslandContext.mThreadContext;

				//Compact articulation pairs...
				ArticulationSolverDesc* artics = mThreadContext.getArticulations().begin();

				if (mIslandContext.mCounts.articulations)
				{
					PxU32 nbArticConstraints = artics[0].numInternalConstraints;

					PxSolverConstraintDesc* currDesc = mThreadContext.mContactDescPtr;
					for (PxU32 a = 1; a < mIslandContext.mCounts.articulations; ++a)
					{
						//Compact pairs...
						const PxU32 nbInternalConstraints = artics[a].numInternalConstraints;
						const PxU32 startIdx = a * DY_ARTICULATION_MAX_SIZE;
						const PxU32 endIdx = startIdx + nbInternalConstraints;

						for (PxU32 b = startIdx; b < endIdx; ++b)
						{
							currDesc[nbArticConstraints++] = currDesc[b];
						}
					}

					mThreadContext.contactDescArraySize += nbArticConstraints;
				}

				PxSolverConstraintDesc* descBegin = mThreadContext.contactConstraintDescArray;
				PxU32 descCount = mThreadContext.contactDescArraySize;

				PxSolverBody* solverBodies = mContext.mSolverBodyPool.begin() + mSolverBodyOffset;

				mThreadContext.mNumDifferentBodyConstraints = descCount;

				{
					mThreadContext.mNumDifferentBodyConstraints = 0;
					mThreadContext.mNumSelfConstraints = 0;
					mThreadContext.mNumStaticConstraints = 0;
					mThreadContext.mNumDifferentBodyFrictionConstraints = 0;
					mThreadContext.mNumSelfConstraintFrictionBlocks = 0;
					mThreadContext.mNumSelfFrictionConstraints = 0;

					if (descCount > 0)
					{
						ConstraintPartitionArgs args;
						args.mBodies = reinterpret_cast<PxU8*>(solverBodies);
						args.mStride = sizeof(PxSolverBody);
						args.mArticulationPtrs = artics;
						args.mContactConstraintDescriptors = descBegin;
						args.mNumArticulationPtrs = mThreadContext.getArticulations().size();
						args.mNumBodies = mIslandContext.mCounts.bodies;
						args.mNumContactConstraintDescriptors = descCount;
						args.mOrderedContactConstraintDescriptors = mThreadContext.orderedContactConstraints;
						args.mTempContactConstraintDescriptors = mThreadContext.tempConstraintDescArray;
						args.mNumDifferentBodyConstraints = args.mNumSelfConstraints = args.mNumStaticConstraints = 0;
						args.mConstraintsPerPartition = &mThreadContext.mConstraintsPerPartition;
						args.mBitField = &mThreadContext.mPartitionNormalizationBitmap;
						args.enhancedDeterminism = mEnhancedDeterminism;

						mThreadContext.mMaxPartitions = partitionContactConstraints(args);
						mThreadContext.mNumDifferentBodyConstraints = args.mNumDifferentBodyConstraints;
						mThreadContext.mNumSelfConstraints = args.mNumSelfConstraints;
						mThreadContext.mNumStaticConstraints = args.mNumStaticConstraints;
					}
					else
					{
						PxMemZero(mThreadContext.mConstraintsPerPartition.begin(), sizeof(PxU32) * mThreadContext.mConstraintsPerPartition.capacity());
					}

					PX_ASSERT((mThreadContext.mNumDifferentBodyConstraints + mThreadContext.mNumSelfConstraints + mThreadContext.mNumStaticConstraints) == descCount);
				}
			}

			virtual const char* getName() const { return "PxsDynamics.solverConstraintPartition"; }

			DynamicsLDLContext& mContext;
			IslandContext& mIslandContext;
			const SolverIslandObjects	mObjects;
			PxU32						mSolverBodyOffset;
			bool						mEnhancedDeterminism;
		};
		
		class PxsSolverConstraintPostProcessTask : public Cm::Task
		{
			PxsSolverConstraintPostProcessTask& operator=(const PxsSolverConstraintPostProcessTask&);
		public:

			PxsSolverConstraintPostProcessTask(DynamicsLDLContext& context,
				ThreadContext& threadContext,
				const SolverIslandObjects& objects,
				const PxU32 solverBodyOffset,
				PxU32 startIndex,
				PxU32 stride,
				PxsMaterialManager* materialManager,
				PxsContactManagerOutputIterator& iterator) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mThreadContext(threadContext),
				mObjects(objects),
				mSolverBodyOffset(solverBodyOffset),
				mStartIndex(startIndex),
				mStride(stride),
				mMaterialManager(materialManager),
				mOutputs(iterator)
			{}

			void mergeContacts(CompoundContactManager& header, ThreadContext& threadContext)
			{
				Gu::ContactBuffer& buffer = threadContext.mContactBuffer;
				PxsMaterialInfo materialInfo[Gu::ContactBuffer::MAX_CONTACTS];
				PxU32 size = 0;

				for (PxU32 a = 0; a < header.mStride; ++a)
				{
					PxsContactManager* manager = mThreadContext.orderedContactList[a + header.mStartIndex]->contactManager;
					PxcNpWorkUnit& unit = manager->getWorkUnit();
					PxsContactManagerOutput& output = mOutputs.getContactManager(unit.mNpIndex);
					PxContactStreamIterator iter(output.contactPatches, output.contactPoints, output.getInternalFaceIndice(), output.nbPatches, output.nbContacts);

					PxU32 origSize = size;
					PX_UNUSED(origSize);
					if (!iter.forceNoResponse)
					{
						while (iter.hasNextPatch())
						{
							iter.nextPatch();
							while (iter.hasNextContact())
							{
								PX_ASSERT(size < Gu::ContactBuffer::MAX_CONTACTS);
								iter.nextContact();
								PxsMaterialInfo& info = materialInfo[size];
								Gu::ContactPoint& point = buffer.contacts[size++];
								point.dynamicFriction = iter.getDynamicFriction();
								point.staticFriction = iter.getStaticFriction();
								point.restitution = iter.getRestitution();
								point.internalFaceIndex1 = iter.getFaceIndex1();
								point.materialFlags = PxU8(iter.getMaterialFlags());
								point.maxImpulse = iter.getMaxImpulse();
								point.targetVel = iter.getTargetVel();
								point.normal = iter.getContactNormal();
								point.point = iter.getContactPoint();
								point.separation = iter.getSeparation();
								info.mMaterialIndex0 = iter.getMaterialIndex0();
								info.mMaterialIndex1 = iter.getMaterialIndex1();
							}
						}
						PX_ASSERT(output.nbContacts == (size - origSize));
					}
				}

				PxU32 origSize = size;
				PX_UNUSED(origSize);
#if PX_CONTACT_REDUCTION
				ContactReduction<6> reduction(buffer.contacts, materialInfo, size);
				reduction.reduceContacts();
				//OK, now we write back the contacts...

				PxU8 histo[Gu::ContactBuffer::MAX_CONTACTS];
				PxMemZero(histo, sizeof(histo));

				size = 0;
				for (PxU32 a = 0; a < reduction.mNumPatches; ++a)
				{
					ReducedContactPatch& patch = reduction.mPatches[a];
					for (PxU32 b = 0; b < patch.numContactPoints; ++b)
					{
						histo[patch.contactPoints[b]] = 1;
						++size;
					}
				}
#endif

				PxU16* PX_RESTRICT data = reinterpret_cast<PxU16*>(threadContext.mConstraintBlockStream.reserve(size * sizeof(PxU16), mThreadContext.mConstraintBlockManager));
				header.forceBufferList = data;

#if PX_CONTACT_REDUCTION
				const PxU32 reservedSize = size;
				PX_UNUSED(reservedSize);
				size = 0;
				for (PxU32 a = 0; a < origSize; ++a)
				{
					if (histo[a])
					{
						if (size != a)
						{
							buffer.contacts[size] = buffer.contacts[a];
							materialInfo[size] = materialInfo[a];
						}
						data[size] = Ps::to16(a);
						size++;
					}
				}
				PX_ASSERT(reservedSize >= size);
#else
				for (PxU32 a = 0; a < size; ++a)
					data[a] = static_cast<PxU16>(a);
#endif

				PxU32 contactForceByteSize = size * sizeof(PxReal);

				PxsContactManagerOutput& output = mOutputs.getContactManager(header.unit->mNpIndex);

				PxU16 compressedContactSize;

				physx::writeCompressedContact(buffer.contacts, size, NULL, output.nbContacts, output.contactPatches, output.contactPoints, compressedContactSize,
					reinterpret_cast<PxReal*&>(output.contactForces), contactForceByteSize, mMaterialManager, false,
					false, materialInfo, output.nbPatches, 0, &mThreadContext.mConstraintBlockManager, &threadContext.mConstraintBlockStream, false);
			}

			virtual void runInternal()
			{
				PX_PROFILE_ZONE("ConstraintPostProcess", mContext.getContextId());
				PxU32 endIndex = mStartIndex + mStride;

				ThreadContext* threadContext = mContext.getThreadContext();
				//TODO - we need to do this somewhere else
				//threadContext->mContactBlockStream.reset();
				threadContext->mConstraintBlockStream.reset();

				for (PxU32 a = mStartIndex; a < endIndex; ++a)
				{
					mergeContacts(mThreadContext.compoundConstraints[a], *threadContext);
				}
				mContext.putThreadContext(threadContext);
			}

			virtual const char* getName() const { return "PxsDynamics.solverConstraintPostProcess"; }

			DynamicsLDLContext& mContext;
			ThreadContext& mThreadContext;
			const SolverIslandObjects	mObjects;
			PxU32						mSolverBodyOffset;
			PxU32						mStartIndex;
			PxU32						mStride;
			PxsMaterialManager* mMaterialManager;
			PxsContactManagerOutputIterator& mOutputs;
		};

		class PxsSolverStartTask : public Cm::Task
		{
			PxsSolverStartTask& operator=(const PxsSolverStartTask&);
		public:

			PxsSolverStartTask(DynamicsLDLContext& context,
				IslandContext& islandContext,
				const SolverIslandObjects& objects,
				const PxU32 solverBodyOffset,
				const PxU32 kinematicCount,
				IG::SimpleIslandManager& islandManager,
				PxU32* bodyRemapTable,
				PxsMaterialManager* materialManager,
				PxsContactManagerOutputIterator& iterator,
				bool enhancedDeterminism
			) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mIslandContext(islandContext),
				mObjects(objects),
				mSolverBodyOffset(solverBodyOffset),
				mKinematicCount(kinematicCount),
				mIslandManager(islandManager),
				mBodyRemapTable(bodyRemapTable),
				mMaterialManager(materialManager),
				mOutputs(iterator),
				mEnhancedDeterminism(enhancedDeterminism)
			{}

			void startTasks()
			{
				PX_PROFILE_ZONE("Dynamics.solveGroup", mContext.getContextId());
				{
					ThreadContext& mThreadContext = *mContext.getThreadContext();

					mIslandContext.mThreadContext = &mThreadContext;

					mThreadContext.mMaxSolverPositionIterations = 0;
					mThreadContext.mMaxSolverVelocityIterations = 0;
					mThreadContext.mAxisConstraintCount = 0;
					mThreadContext.mContactDescPtr = mThreadContext.contactConstraintDescArray;
					mThreadContext.mFrictionDescPtr = mThreadContext.frictionConstraintDescArray.begin();
					mThreadContext.mNumDifferentBodyConstraints = 0;
					mThreadContext.mNumStaticConstraints = 0;
					mThreadContext.mNumSelfConstraints = 0;
					mThreadContext.mNumDifferentBodyFrictionConstraints = 0;
					mThreadContext.mNumSelfConstraintFrictionBlocks = 0;
					mThreadContext.mNumSelfFrictionConstraints = 0;
					mThreadContext.numContactConstraintBatches = 0;
					mThreadContext.contactDescArraySize = 0;
					mThreadContext.mMaxArticulationLinks = 0;

					mThreadContext.contactConstraintDescArray = mObjects.constraintDescs;
					mThreadContext.orderedContactConstraints = mObjects.orderedConstraintDescs;
					mThreadContext.mContactDescPtr = mObjects.constraintDescs;
					mThreadContext.tempConstraintDescArray = mObjects.tempConstraintDescs;
					mThreadContext.contactConstraintBatchHeaders = mObjects.constraintBatchHeaders;
					mThreadContext.motionVelocityArray = mObjects.motionVelocities;
					mThreadContext.mBodyCoreArray = mObjects.bodyCoreArray;
					mThreadContext.mRigidBodyArray = mObjects.bodies;
					mThreadContext.mArticulationArray = mObjects.articulations;
					mThreadContext.bodyRemapTable = mObjects.bodyRemapTable;
					mThreadContext.mNodeIndexArray = mObjects.nodeIndexArray;

					const PxU32 frictionConstraintCount = mContext.getFrictionType() == PxFrictionType::ePATCH ? 0 : PxU32(mIslandContext.mCounts.contactManagers);
					mThreadContext.resizeArrays(frictionConstraintCount, mIslandContext.mCounts.articulations);

					PxsBodyCore** PX_RESTRICT bodyArrayPtr = mThreadContext.mBodyCoreArray;
					PxsRigidBody** PX_RESTRICT rigidBodyPtr = mThreadContext.mRigidBodyArray;
					ArticulationV** PX_RESTRICT articulationPtr = mThreadContext.mArticulationArray;
					PxU32* PX_RESTRICT bodyRemapTable = mThreadContext.bodyRemapTable;
					PxU32* PX_RESTRICT nodeIndexArray = mThreadContext.mNodeIndexArray;

					PxU32 nbIslands = mObjects.numIslands;
					const IG::IslandId* const islandIds = mObjects.islandIds;

					const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();

					PxU32 bodyIndex = 0, articIndex = 0;
					for (PxU32 i = 0; i < nbIslands; ++i)
					{
						const IG::Island& island = islandSim.getIsland(islandIds[i]);

						IG::NodeIndex currentIndex = island.mRootNode;

						while (currentIndex.isValid())
						{
							const IG::Node& node = islandSim.getNode(currentIndex);

							if (node.getNodeType() == IG::Node::eARTICULATION_TYPE)
							{
								articulationPtr[articIndex++] = node.getArticulation();
							}
							else
							{
								PX_ASSERT(bodyIndex < (mIslandContext.mCounts.bodies + mContext.mKinematicCount + 1));
								nodeIndexArray[bodyIndex++] = currentIndex.index();
							}

							currentIndex = node.mNextNode;
						}
					}

					//Bodies can come in a slightly jumbled order from islandGen. It's deterministic if the scene is 
					//identical but can vary if there are additional bodies in the scene in a different island.
					if (mEnhancedDeterminism)
					{
						Ps::sort(nodeIndexArray, bodyIndex);
					}

					for (PxU32 a = 0; a < bodyIndex; ++a)
					{
						IG::NodeIndex currentIndex(nodeIndexArray[a]);
						const IG::Node& node = islandSim.getNode(currentIndex);
						PxsRigidBody* rigid = node.getRigidBody();
						rigidBodyPtr[a] = rigid;
						bodyArrayPtr[a] = &rigid->getCore();
						bodyRemapTable[islandSim.getActiveNodeIndex(currentIndex)] = a;
					}

					PxsIndexedContactManager* indexedManagers = mObjects.contactManagers;

					PxU32 currentContactIndex = 0;
					for (PxU32 i = 0; i < nbIslands; ++i)
					{
						const IG::Island& island = islandSim.getIsland(islandIds[i]);

						IG::EdgeIndex contactEdgeIndex = island.mFirstEdge[IG::Edge::eCONTACT_MANAGER];

						while (contactEdgeIndex != IG_INVALID_EDGE)
						{
							const IG::Edge& edge = islandSim.getEdge(contactEdgeIndex);

							PxsContactManager* contactManager = mIslandManager.getContactManager(contactEdgeIndex);

							if (contactManager)
							{
								const IG::NodeIndex nodeIndex1 = islandSim.getNodeIndex1(contactEdgeIndex);
								const IG::NodeIndex nodeIndex2 = islandSim.getNodeIndex2(contactEdgeIndex);

								PxsIndexedContactManager& indexedManager = indexedManagers[currentContactIndex++];
								indexedManager.contactManager = contactManager;

								PX_ASSERT(!nodeIndex1.isStaticBody());
								{
									const IG::Node& node1 = islandSim.getNode(nodeIndex1);

									//Is it an articulation or not???
									if (node1.getNodeType() == IG::Node::eARTICULATION_TYPE)
									{
										const PxU32 linkId = nodeIndex1.articulationLinkId();
										node1.getArticulation()->fillIndexedManager(linkId, indexedManager.articulation0, indexedManager.indexType0);
									}
									else
									{
										if (node1.isKinematic())
										{
											indexedManager.indexType0 = PxsIndexedInteraction::eKINEMATIC;
											indexedManager.solverBody0 = islandSim.getActiveNodeIndex(nodeIndex1);
										}
										else
										{
											indexedManager.indexType0 = PxsIndexedInteraction::eBODY;
											indexedManager.solverBody0 = bodyRemapTable[islandSim.getActiveNodeIndex(nodeIndex1)];
										}
										PX_ASSERT(indexedManager.solverBody0 < (mIslandContext.mCounts.bodies + mContext.mKinematicCount + 1));
									}

								}

								if (nodeIndex2.isStaticBody())
								{
									indexedManager.indexType1 = PxsIndexedInteraction::eWORLD;
								}
								else
								{
									const IG::Node& node2 = islandSim.getNode(nodeIndex2);

									//Is it an articulation or not???
									if (node2.getNodeType() == IG::Node::eARTICULATION_TYPE)
									{
										const PxU32 linkId = nodeIndex2.articulationLinkId();
										node2.getArticulation()->fillIndexedManager(linkId, indexedManager.articulation1, indexedManager.indexType1);
									}
									else
									{
										if (node2.isKinematic())
										{
											indexedManager.indexType1 = PxsIndexedInteraction::eKINEMATIC;
											indexedManager.solverBody1 = islandSim.getActiveNodeIndex(nodeIndex2);
										}
										else
										{
											indexedManager.indexType1 = PxsIndexedInteraction::eBODY;
											indexedManager.solverBody1 = bodyRemapTable[islandSim.getActiveNodeIndex(nodeIndex2)];
										}
										PX_ASSERT(indexedManager.solverBody1 < (mIslandContext.mCounts.bodies + mContext.mKinematicCount + 1));
									}
								}

							}
							contactEdgeIndex = edge.mNextIslandEdge;
						}
					}

					if (mEnhancedDeterminism)
					{
						Ps::sort(indexedManagers, currentContactIndex, EnhancedSortPredicate());
					}

					mIslandContext.mCounts.contactManagers = currentContactIndex;
				}
			}

			void integrate()
			{
				ThreadContext& mThreadContext = *mIslandContext.mThreadContext;
				PxSolverBody* solverBodies = mContext.mSolverBodyPool.begin() + mSolverBodyOffset;
				PxSolverBodyData* solverBodyData = mContext.mSolverBodyDataPool.begin() + mSolverBodyOffset;

				{
					PX_PROFILE_ZONE("Dynamics.updateVelocities", mContext.getContextId());

					mContext.preIntegrationParallel(
						mContext.mDt,
						mThreadContext.mBodyCoreArray,
						mObjects.bodies,
						mThreadContext.mNodeIndexArray,
						mIslandContext.mCounts.bodies,
						solverBodies,
						solverBodyData,
						mThreadContext.motionVelocityArray,
						mThreadContext.mMaxSolverPositionIterations,
						mThreadContext.mMaxSolverVelocityIterations,
						*mCont
					);
				}
			}

			void articulationTask()
			{
				ThreadContext& mThreadContext = *mIslandContext.mThreadContext;
				ArticulationSolverDesc* articulationDescArray = mThreadContext.getArticulations().begin();

				for (PxU32 i = 0; i < mIslandContext.mCounts.articulations; i += SolverArticulationUpdateTask::NbArticulationsPerTask)
				{
					SolverArticulationUpdateTask* task = PX_PLACEMENT_NEW(mContext.getTaskPool().allocate(sizeof(SolverArticulationUpdateTask)), SolverArticulationUpdateTask)(mThreadContext,
						&mObjects.articulations[i], &articulationDescArray[i], PxMin(SolverArticulationUpdateTask::NbArticulationsPerTask, mIslandContext.mCounts.articulations - i), mContext,
						i * DY_ARTICULATION_MAX_SIZE);

					task->setContinuation(mCont);
					task->removeReference();
				}
			}

			void setupDescTask()
			{
				PX_PROFILE_ZONE("SetupDescs", mContext.getContextId());
				ThreadContext& mThreadContext = *mIslandContext.mThreadContext;
				PxSolverConstraintDesc* contactDescPtr = mThreadContext.mContactDescPtr;

				//PxU32 constraintCount = mCounts.constraints + mCounts.contactManagers;

				PxU32 nbIslands = mObjects.numIslands;
				const IG::IslandId* const islandIds = mObjects.islandIds;

				const IG::IslandSim& islandSim = mIslandManager.getAccurateIslandSim();

				for (PxU32 i = 0; i < nbIslands; ++i)
				{
					const IG::Island& island = islandSim.getIsland(islandIds[i]);

					IG::EdgeIndex edgeId = island.mFirstEdge[IG::Edge::eCONSTRAINT];

					while (edgeId != IG_INVALID_EDGE)
					{
						PxSolverConstraintDesc& desc = *contactDescPtr;

						const IG::Edge& edge = islandSim.getEdge(edgeId);
						Dy::Constraint* constraint = mIslandManager.getConstraint(edgeId);
						mContext.setDescFromIndices(desc, edgeId, mIslandManager, mBodyRemapTable, mSolverBodyOffset);
						desc.constraint = reinterpret_cast<PxU8*>(constraint);
						desc.constraintLengthOver16 = DY_SC_TYPE_RB_1D;
						contactDescPtr++;
						edgeId = edge.mNextIslandEdge;
					}

				}

#if 1
				Ps::sort(mThreadContext.mContactDescPtr, PxU32(contactDescPtr - mThreadContext.mContactDescPtr), ConstraintLess());
#endif

				mThreadContext.orderedContactList.forceSize_Unsafe(0);
				mThreadContext.orderedContactList.reserve(mIslandContext.mCounts.contactManagers);
				mThreadContext.orderedContactList.forceSize_Unsafe(mIslandContext.mCounts.contactManagers);
				mThreadContext.tempContactList.forceSize_Unsafe(0);
				mThreadContext.tempContactList.reserve(mIslandContext.mCounts.contactManagers);
				mThreadContext.tempContactList.forceSize_Unsafe(mIslandContext.mCounts.contactManagers);

				const PxsIndexedContactManager** constraints = mThreadContext.orderedContactList.begin();

				//OK, we sort the orderedContactList 

				mThreadContext.compoundConstraints.forceSize_Unsafe(0);
				if (mIslandContext.mCounts.contactManagers)
				{
					{
						mThreadContext.sortIndexArray.forceSize_Unsafe(0);

						PX_COMPILE_TIME_ASSERT(PxsIndexedInteraction::eBODY == 0);
						PX_COMPILE_TIME_ASSERT(PxsIndexedInteraction::eKINEMATIC == 1);

						const PxI32 offsetMap[] = { PxI32(mContext.mKinematicCount), 0 };

						const PxU32 totalBodies = mContext.mKinematicCount + mIslandContext.mCounts.bodies + 1;

						mThreadContext.sortIndexArray.reserve(totalBodies);
						mThreadContext.sortIndexArray.forceSize_Unsafe(totalBodies);
						PxMemZero(mThreadContext.sortIndexArray.begin(), totalBodies * 4);

						//Iterate over the array based on solverBodyDatapool, creating a list of sorted constraints (in order of body pair)
						//We only do this with contacts. It's important that this is done this way because we don't want to break our rules that all joints
						//appear before all contacts in the constraint list otherwise we will lose all guarantees about sorting joints.

						for (PxU32 a = 0; a < mIslandContext.mCounts.contactManagers; ++a)
						{
							PX_ASSERT(mObjects.contactManagers[a].indexType0 != PxsIndexedInteraction::eWORLD);
							//Index first body...
							PxU8 indexType = mObjects.contactManagers[a].indexType0;
							if (indexType != PxsIndexedInteraction::eARTICULATION && mObjects.contactManagers[a].indexType1 != PxsIndexedInteraction::eARTICULATION)
							{
								PX_ASSERT((indexType == PxsIndexedInteraction::eBODY) || (indexType == PxsIndexedInteraction::eKINEMATIC));

								PxI32 index = PxI32(mObjects.contactManagers[a].solverBody0 + offsetMap[indexType]);
								PX_ASSERT(index >= 0);
								mThreadContext.sortIndexArray[PxU32(index)]++;
							}
						}

						PxU32 accumulatedCount = 0;

						for (PxU32 a = mThreadContext.sortIndexArray.size(); a > 0; --a)
						{
							PxU32 ind = a - 1;
							PxU32 val = mThreadContext.sortIndexArray[ind];
							mThreadContext.sortIndexArray[ind] = accumulatedCount;
							accumulatedCount += val;
						}

						//OK, now copy across data to orderedConstraintDescs, pushing articulations to the end...
						for (PxU32 a = 0; a < mIslandContext.mCounts.contactManagers; ++a)
						{
							//Index first body...
							PxU8 indexType = mObjects.contactManagers[a].indexType0;
							if (indexType != PxsIndexedInteraction::eARTICULATION && mObjects.contactManagers[a].indexType1 != PxsIndexedInteraction::eARTICULATION)
							{
								PX_ASSERT((indexType == PxsIndexedInteraction::eBODY) || (indexType == PxsIndexedInteraction::eKINEMATIC));

								PxI32 index = PxI32(mObjects.contactManagers[a].solverBody0 + offsetMap[indexType]);
								PX_ASSERT(index >= 0);
								mThreadContext.tempContactList[mThreadContext.sortIndexArray[PxU32(index)]++] = &mObjects.contactManagers[a];
							}
							else
							{
								mThreadContext.tempContactList[accumulatedCount++] = &mObjects.contactManagers[a];
							}
						}

						//Now do the same again with bodyB, being careful not to overwrite the joints
						PxMemZero(mThreadContext.sortIndexArray.begin(), totalBodies * 4);

						for (PxU32 a = 0; a < mIslandContext.mCounts.contactManagers; ++a)
						{
							//Index first body...
							PxU8 indexType = mThreadContext.tempContactList[a]->indexType1;
							if (indexType != PxsIndexedInteraction::eARTICULATION && mObjects.contactManagers[a].indexType0 != PxsIndexedInteraction::eARTICULATION)
							{
								PX_ASSERT((indexType == PxsIndexedInteraction::eBODY) || (indexType == PxsIndexedInteraction::eKINEMATIC) || (indexType == PxsIndexedInteraction::eWORLD));

								PxI32 index = (indexType == PxsIndexedInteraction::eWORLD) ? 0 : PxI32(mThreadContext.tempContactList[a]->solverBody1 + offsetMap[indexType]);
								PX_ASSERT(index >= 0);
								mThreadContext.sortIndexArray[PxU32(index)]++;
							}
						}

						accumulatedCount = 0;
						for (PxU32 a = mThreadContext.sortIndexArray.size(); a > 0; --a)
						{
							PxU32 ind = a - 1;
							PxU32 val = mThreadContext.sortIndexArray[ind];
							mThreadContext.sortIndexArray[ind] = accumulatedCount;
							accumulatedCount += val;
						}

						PxU32 articulationStartIndex = accumulatedCount;

						//OK, now copy across data to orderedConstraintDescs, pushing articulations to the end...
						for (PxU32 a = 0; a < mIslandContext.mCounts.contactManagers; ++a)
						{
							//Index first body...
							PxU8 indexType = mThreadContext.tempContactList[a]->indexType1;
							if (indexType != PxsIndexedInteraction::eARTICULATION && mObjects.contactManagers[a].indexType0 != PxsIndexedInteraction::eARTICULATION)
							{
								PX_ASSERT((indexType == PxsIndexedInteraction::eBODY) || (indexType == PxsIndexedInteraction::eKINEMATIC) || (indexType == PxsIndexedInteraction::eWORLD));

								PxI32 index = (indexType == PxsIndexedInteraction::eWORLD) ? 0 : PxI32(mThreadContext.tempContactList[a]->solverBody1 + offsetMap[indexType]);
								PX_ASSERT(index >= 0);
								constraints[mThreadContext.sortIndexArray[PxU32(index)]++] = mThreadContext.tempContactList[a];
							}
							else
							{
								constraints[accumulatedCount++] = mThreadContext.tempContactList[a];
							}
						}

#if 1
						Ps::sort(constraints + articulationStartIndex, accumulatedCount - articulationStartIndex, ArticulationSortPredicate());
#endif
					}

					mThreadContext.mStartContactDescPtr = contactDescPtr;

					mThreadContext.compoundConstraints.reserve(1024);
					mThreadContext.compoundConstraints.forceSize_Unsafe(0);
					//mThreadContext.compoundConstraints.forceSize_Unsafe(mCounts.contactManagers);

					PxSolverConstraintDesc* startDesc = contactDescPtr;
					mContext.setDescFromIndices(*startDesc, *constraints[0], mSolverBodyOffset);
					startDesc->constraint = reinterpret_cast<PxU8*>(constraints[0]->contactManager);
					startDesc->constraintLengthOver16 = DY_SC_TYPE_RB_CONTACT;

					PxsContactManagerOutput* startManagerOutput = &mOutputs.getContactManager(constraints[0]->contactManager->getWorkUnit().mNpIndex);
					PxU32 contactCount = startManagerOutput->nbContacts;
					PxU32 startIndex = 0;
					PxU32 numHeaders = 0;

					const bool gDisableConstraintWelding = false;

					for (PxU32 a = 1; a < mIslandContext.mCounts.contactManagers; ++a)
					{
						PxSolverConstraintDesc& desc = *(contactDescPtr + 1);
						mContext.setDescFromIndices(desc, *constraints[a], mSolverBodyOffset);

						PxsContactManager* manager = constraints[a]->contactManager;
						PxsContactManagerOutput& output = mOutputs.getContactManager(manager->getWorkUnit().mNpIndex);

						desc.constraint = reinterpret_cast<PxU8*>(constraints[a]->contactManager);
						desc.constraintLengthOver16 = DY_SC_TYPE_RB_CONTACT;

						if (contactCount == 0)
						{
							//This is the first object in the pair
							*startDesc = *(contactDescPtr + 1);
							startIndex = a;
							startManagerOutput = &output;
						}

						if (startDesc->bodyA != desc.bodyA || startDesc->bodyB != desc.bodyB
							|| startDesc->linkIndexA != PxSolverConstraintDesc::NO_LINK || startDesc->linkIndexB != PxSolverConstraintDesc::NO_LINK
							|| contactCount + output.nbContacts > Gu::ContactBuffer::MAX_CONTACTS
							|| manager->isChangeable()
							|| gDisableConstraintWelding
							) //If this is the first thing and no contacts...then we skip
						{
							PxU32 stride = a - startIndex;
							if (contactCount > 0)
							{
								if (stride > 1)
								{
									++numHeaders;
									CompoundContactManager& header = mThreadContext.compoundConstraints.insert();
									header.mStartIndex = startIndex;
									header.mStride = Ps::to16(stride);
									header.mReducedContactCount = Ps::to16(contactCount);
									PxsContactManager* manager1 = constraints[startIndex]->contactManager;
									PxcNpWorkUnit& unit = manager1->getWorkUnit();

									PX_ASSERT(startManagerOutput == &mOutputs.getContactManager(unit.mNpIndex));

									header.unit = &unit;
									header.cmOutput = startManagerOutput;
									header.originalContactPatches = startManagerOutput->contactPatches;
									header.originalContactPoints = startManagerOutput->contactPoints;
									header.originalContactCount = startManagerOutput->nbContacts;
									header.originalPatchCount = startManagerOutput->nbPatches;
									header.originalForceBuffer = reinterpret_cast<PxReal*>(startManagerOutput->contactForces);
									header.originalStatusFlags = startManagerOutput->statusFlag;
								}
								startDesc = ++contactDescPtr;
							}
							else
							{
								//Copy back next contactDescPtr
								*startDesc = *(contactDescPtr + 1);
							}
							contactCount = 0;
							startIndex = a;
							startManagerOutput = &output;
						}
						contactCount += output.nbContacts;

					}
					PxU32 stride = mIslandContext.mCounts.contactManagers - startIndex;
					if (contactCount > 0)
					{
						if (stride > 1)
						{
							++numHeaders;
							CompoundContactManager& header = mThreadContext.compoundConstraints.insert();
							header.mStartIndex = startIndex;
							header.mStride = Ps::to16(stride);
							header.mReducedContactCount = Ps::to16(contactCount);
							PxsContactManager* manager = constraints[startIndex]->contactManager;
							PxcNpWorkUnit& unit = manager->getWorkUnit();
							header.unit = &unit;
							header.cmOutput = startManagerOutput;
							header.originalContactPatches = startManagerOutput->contactPatches;
							header.originalContactPoints = startManagerOutput->contactPoints;
							header.originalContactCount = startManagerOutput->nbContacts;
							header.originalPatchCount = startManagerOutput->nbPatches;
							header.originalForceBuffer = reinterpret_cast<PxReal*>(startManagerOutput->contactForces);
							header.originalStatusFlags = startManagerOutput->statusFlag;
						}
						contactDescPtr++;
					}

					if (numHeaders)
					{
						const PxU32 unrollSize = 8;
						for (PxU32 a = 0; a < numHeaders; a += unrollSize)
						{
							PxsSolverConstraintPostProcessTask* postProcessTask = PX_PLACEMENT_NEW(mContext.getTaskPool().allocate(sizeof(PxsSolverConstraintPostProcessTask)),
								PxsSolverConstraintPostProcessTask)(mContext, mThreadContext, mObjects, mSolverBodyOffset, a, PxMin(unrollSize, numHeaders - a), mMaterialManager,
									mOutputs);
							postProcessTask->setContinuation(mCont);
							postProcessTask->removeReference();
						}
					}
				}
				mThreadContext.contactDescArraySize = PxU32(contactDescPtr - mThreadContext.contactConstraintDescArray);
				mThreadContext.mContactDescPtr = contactDescPtr;
			}

			virtual void runInternal()
			{
				startTasks();
				integrate();
				setupDescTask();
				articulationTask();
			}

			virtual const char* getName() const
			{
				return "PxsDynamics.solverStart";
			}

		private:
			DynamicsLDLContext& mContext;
			IslandContext& mIslandContext;
			const SolverIslandObjects	mObjects;
			const PxU32					mSolverBodyOffset;
			const PxU32					mKinematicCount;
			IG::SimpleIslandManager& mIslandManager;
			PxU32* mBodyRemapTable;
			PxsMaterialManager* mMaterialManager;
			PxsContactManagerOutputIterator& mOutputs;
			bool						mEnhancedDeterminism;
		};

		class PxsSolverEndTask : public Cm::Task
		{
			PxsSolverEndTask& operator=(const PxsSolverEndTask&);
		public:

			PxsSolverEndTask(DynamicsLDLContext& context,
				IslandContext& islandContext,
				const SolverIslandObjects& objects,
				const PxU32 solverBodyOffset,
				PxsContactManagerOutputIterator& cmOutputs) :
				Cm::Task(context.getContextId()),
				mContext(context),
				mIslandContext(islandContext),
				mObjects(objects),
				mSolverBodyOffset(solverBodyOffset),
				mOutputs(cmOutputs)
			{}

			virtual void runInternal()
			{
				PX_PROFILE_ZONE("Dynamics.endTask", getContextId());
				ThreadContext& mThreadContext = *mIslandContext.mThreadContext;
#if PX_ENABLE_SIM_STATS
				mThreadContext.getSimStats().numAxisSolverConstraints += mThreadContext.mAxisConstraintCount;
#endif
				//Patch up the contact managers (TODO - fix up force writeback)
				PxU32 numCompoundConstraints = mThreadContext.compoundConstraints.size();
				for (PxU32 i = 0; i < numCompoundConstraints; ++i)
				{
					CompoundContactManager& manager = mThreadContext.compoundConstraints[i];
					PxsContactManagerOutput* cmOutput = manager.cmOutput;

					PxReal* contactForces = reinterpret_cast<PxReal*>(cmOutput->contactForces);
					PxU32 contactCount = cmOutput->nbContacts;

					cmOutput->contactPatches = manager.originalContactPatches;
					cmOutput->contactPoints = manager.originalContactPoints;
					cmOutput->nbContacts = manager.originalContactCount;
					cmOutput->nbPatches = manager.originalPatchCount;
					cmOutput->statusFlag = manager.originalStatusFlags;
					cmOutput->contactForces = manager.originalForceBuffer;

					for (PxU32 a = 1; a < manager.mStride; ++a)
					{
						PxsContactManager* pManager = mThreadContext.orderedContactList[manager.mStartIndex + a]->contactManager;
						pManager->getWorkUnit().frictionDataPtr = manager.unit->frictionDataPtr;
						pManager->getWorkUnit().frictionPatchCount = manager.unit->frictionPatchCount;
						//pManager->getWorkUnit().prevFrictionPatchCount = manager.unit->prevFrictionPatchCount;
					}

					//This is a stride-based contact force writer. The assumption is that we may have skipped certain unimportant contacts reported by the 
					//discrete narrow phase
					if (contactForces)
					{
						PxU32 currentContactIndex = 0;
						PxU32 currentManagerIndex = manager.mStartIndex;
						PxU32 currentManagerContactIndex = 0;

						for (PxU32 a = 0; a < contactCount; ++a)
						{
							PxU32 index = manager.forceBufferList[a];
							PxsContactManager* pManager = mThreadContext.orderedContactList[currentManagerIndex]->contactManager;
							PxsContactManagerOutput* output = &mOutputs.getContactManager(pManager->getWorkUnit().mNpIndex);
							while (currentContactIndex < index || output->nbContacts == 0)
							{
								//Step forwards...first in this manager...

								PxU32 numToStep = PxMin(index - currentContactIndex, PxU32(output->nbContacts) - currentManagerContactIndex);
								currentContactIndex += numToStep;
								currentManagerContactIndex += numToStep;
								if (currentManagerContactIndex == output->nbContacts)
								{
									currentManagerIndex++;
									currentManagerContactIndex = 0;
									pManager = mThreadContext.orderedContactList[currentManagerIndex]->contactManager;
									output = &mOutputs.getContactManager(pManager->getWorkUnit().mNpIndex);
								}
							}
							if (output->nbContacts > 0 && output->contactForces)
								output->contactForces[currentManagerContactIndex] = contactForces[a];
						}
					}
				}

				mThreadContext.compoundConstraints.forceSize_Unsafe(0);

				mThreadContext.mConstraintBlockManager.reset();

				mContext.putThreadContext(&mThreadContext);
			}

			virtual const char* getName() const
			{
				return "PxsDynamics.solverEnd";
			}

			DynamicsLDLContext& mContext;
			IslandContext& mIslandContext;
			const SolverIslandObjects			mObjects;
			const PxU32							mSolverBodyOffset;
			PxsContactManagerOutputIterator& mOutputs;
		};

		class UpdateContinuationTask : public Cm::Task
		{
			DynamicsLDLContext& mContext;
			IG::SimpleIslandManager& mSimpleIslandManager;
			PxBaseTask* mLostTouchTask;

			PX_NOCOPY(UpdateContinuationTask)
		public:

			UpdateContinuationTask(DynamicsLDLContext& context,
				IG::SimpleIslandManager& simpleIslandManager,
				PxBaseTask* lostTouchTask,
				PxU64 contextID) : Cm::Task(contextID), mContext(context), mSimpleIslandManager(simpleIslandManager),
				mLostTouchTask(lostTouchTask)
			{
			}

			virtual const char* getName() const { return "UpdateContinuationTask"; }

			virtual void runInternal()
			{
				mContext.updatePostKinematic(mSimpleIslandManager, mCont, mLostTouchTask);
				//Allow lost touch task to run once all tasks have be scheduled
				mLostTouchTask->removeReference();
			}
		};
		
		class KinematicCopyTask : public Cm::Task
		{
			const IG::NodeIndex* const mKinematicIndices;
			const PxU32 mNbKinematics;
			const IG::IslandSim& mIslandSim;
			PxSolverBodyData* mBodyData;

			PX_NOCOPY(KinematicCopyTask)

		public:

			static const PxU32 NbKinematicsPerTask = 1024;

			KinematicCopyTask(const IG::NodeIndex* const kinematicIndices,
				const PxU32 nbKinematics, const IG::IslandSim& islandSim,
				PxSolverBodyData* datas, PxU64 contextID) : Cm::Task(contextID),
				mKinematicIndices(kinematicIndices), mNbKinematics(nbKinematics),
				mIslandSim(islandSim), mBodyData(datas)
			{
			}

			virtual const char* getName() const { return "KinematicCopyTask"; }

			virtual void runInternal()
			{
				for (PxU32 i = 0; i < mNbKinematics; i++)
				{
					PxsRigidBody* rigidBody = mIslandSim.getRigidBody(mKinematicIndices[i]);
					const PxsBodyCore& core = rigidBody->getCore();
					copyToSolverBodyData(core.linearVelocity, core.angularVelocity, core.inverseMass, core.inverseInertia, core.body2World, core.maxPenBias,
						core.maxContactImpulse, mKinematicIndices[i].index(), core.contactReportThreshold, mBodyData[i + 1], core.lockFlags);
					rigidBody->saveLastCCDTransform();
				}
			}
		};

		class PxsForceThresholdTask : public Cm::Task
		{
			DynamicsLDLContext& mDynamicsContext;

			PxsForceThresholdTask& operator=(const PxsForceThresholdTask&);
		public:

			PxsForceThresholdTask(DynamicsLDLContext& context) : Cm::Task(context.getContextId()), mDynamicsContext(context)
			{
			}

			void createForceChangeThresholdStream()
			{
				ThresholdStream& thresholdStream = mDynamicsContext.getThresholdStream();
				//bool haveThresholding = thresholdStream.size()!=0;

				ThresholdTable& thresholdTable = mDynamicsContext.getThresholdTable();
				thresholdTable.build(thresholdStream);

				//generate current force exceeded threshold stream
				ThresholdStream& curExceededForceThresholdStream = *mDynamicsContext.mExceededForceThresholdStream[mDynamicsContext.mCurrentIndex];
				ThresholdStream& preExceededForceThresholdStream = *mDynamicsContext.mExceededForceThresholdStream[1 - mDynamicsContext.mCurrentIndex];
				curExceededForceThresholdStream.forceSize_Unsafe(0);

				//fill in the currrent exceeded force threshold stream
				for (PxU32 i = 0; i < thresholdTable.mPairsSize; ++i)
				{
					ThresholdTable::Pair& pair = thresholdTable.mPairs[i];
					ThresholdStreamElement& elem = thresholdStream[pair.thresholdStreamIndex];
					if (pair.accumulatedForce > elem.threshold * mDynamicsContext.mDt)
					{
						elem.accumulatedForce = pair.accumulatedForce;
						curExceededForceThresholdStream.pushBack(elem);
					}
				}

				ThresholdStream& forceChangeThresholdStream = mDynamicsContext.getForceChangedThresholdStream();
				forceChangeThresholdStream.forceSize_Unsafe(0);
				Ps::Array<PxU32>& forceChangeMask = mDynamicsContext.mExceededForceThresholdStreamMask;

				const PxU32 nbPreExceededForce = preExceededForceThresholdStream.size();
				const PxU32 nbCurExceededForce = curExceededForceThresholdStream.size();

				//generate force change thresholdStream
				if (nbPreExceededForce)
				{
					thresholdTable.build(preExceededForceThresholdStream);

					//set force change mask
					const PxU32 nbTotalExceededForce = nbPreExceededForce + nbCurExceededForce;
					forceChangeMask.reserve(nbTotalExceededForce);
					forceChangeMask.forceSize_Unsafe(nbTotalExceededForce);

					//initialize the forceChangeMask
					for (PxU32 i = 0; i < nbTotalExceededForce; ++i)
						forceChangeMask[i] = 1;

					for (PxU32 i = 0; i < nbCurExceededForce; ++i)
					{
						ThresholdStreamElement& curElem = curExceededForceThresholdStream[i];

						PxU32 pos;
						if (thresholdTable.check(preExceededForceThresholdStream, curElem, pos))
						{
							forceChangeMask[pos] = 0;
							forceChangeMask[i + nbPreExceededForce] = 0;
						}
					}

					//create force change threshold stream
					for (PxU32 i = 0; i < nbTotalExceededForce; ++i)
					{
						const PxU32 hasForceChange = forceChangeMask[i];
						if (hasForceChange)
						{
							bool lostPair = (i < nbPreExceededForce);
							ThresholdStreamElement& elem = lostPair ? preExceededForceThresholdStream[i] : curExceededForceThresholdStream[i - nbPreExceededForce];
							ThresholdStreamElement elt;
							elt = elem;
							elt.accumulatedForce = lostPair ? 0.f : elem.accumulatedForce;
							forceChangeThresholdStream.pushBack(elt);
						}
						else
						{
							//persistent pair
							if (i < nbPreExceededForce)
							{
								ThresholdStreamElement& elem = preExceededForceThresholdStream[i];
								ThresholdStreamElement elt;
								elt = elem;
								elt.accumulatedForce = elem.accumulatedForce;
								forceChangeThresholdStream.pushBack(elt);
							}
						}
					}
				}
				else
				{
					forceChangeThresholdStream.reserve(nbCurExceededForce);
					forceChangeThresholdStream.forceSize_Unsafe(nbCurExceededForce);
					PxMemCopy(forceChangeThresholdStream.begin(), curExceededForceThresholdStream.begin(), sizeof(ThresholdStreamElement) * nbCurExceededForce);
				}
			}

			virtual void runInternal()
			{
				mDynamicsContext.getThresholdStream().forceSize_Unsafe(PxU32(mDynamicsContext.mThresholdStreamOut));
				createForceChangeThresholdStream();
			}

			virtual const char* getName() const { return "PxsDynamics.createForceChangeThresholdStream"; }
		};

		void chainTasksLDL(PxLightCpuTask* first, PxLightCpuTask* next)
		{
			first->setContinuation(next);
			next->removeReference();
		}
	}
}

#endif