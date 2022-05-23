#include "DyDynamicsLDL.h"
#include "DyDynamicsLDLPre.h"

#include "DySolverLDL.h"

namespace physx
{
	namespace Dy
	{
		Context* createDynamicsLDLContext(PxcNpMemBlockPool* memBlockPool,
			PxcScratchAllocator& scratchAllocator, Cm::FlushPool& taskPool,
			PxvSimStats& simStats, PxTaskManager* taskManager, Ps::VirtualAllocatorCallback* allocatorCallback,
			PxsMaterialManager* materialManager, IG::IslandSim* accurateIslandSim, PxU64 contextID,
			const bool enableStabilization, const bool useEnhancedDeterminism, const bool useAdaptiveForce,
			const PxReal maxBiasCoefficient, const bool frictionEveryIteration
		)
		{
			return DynamicsLDLContext::create(memBlockPool, scratchAllocator, taskPool, simStats, taskManager, allocatorCallback, materialManager, accurateIslandSim,
				contextID, enableStabilization, useEnhancedDeterminism, useAdaptiveForce, maxBiasCoefficient, frictionEveryIteration);
		}

		// PT: TODO: consider removing this function. We already have "createDynamicsLDLContext".
		DynamicsLDLContext* DynamicsLDLContext::create(PxcNpMemBlockPool* memBlockPool,
			PxcScratchAllocator& scratchAllocator,
			Cm::FlushPool& taskPool,
			PxvSimStats& simStats,
			PxTaskManager* taskManager,
			Ps::VirtualAllocatorCallback* allocatorCallback,
			PxsMaterialManager* materialManager,
			IG::IslandSim* accurateIslandSim,
			PxU64 contextID,
			const bool enableStabilization,
			const bool useEnhancedDeterminism,
			const bool useAdaptiveForce,
			const PxReal maxBiasCoefficient,
			const bool frictionEveryIteration
		)
		{
			// PT: TODO: inherit from UserAllocated, remove placement new
			DynamicsLDLContext* dc = reinterpret_cast<DynamicsLDLContext*>(PX_ALLOC(sizeof(DynamicsLDLContext), "DynamicsLDLContext"));
			if (dc)
			{
				new(dc)DynamicsLDLContext(memBlockPool, scratchAllocator, taskPool, simStats, taskManager, allocatorCallback, materialManager, accurateIslandSim, contextID,
					enableStabilization, useEnhancedDeterminism, useAdaptiveForce, maxBiasCoefficient, frictionEveryIteration);
			}
			return dc;
		}

		void DynamicsLDLContext::destroy()
		{
			this->~DynamicsLDLContext();
			PX_FREE(this);
		}

		void DynamicsLDLContext::resetThreadContexts()
		{
			PxcThreadCoherentCacheIterator<ThreadContext, PxcNpMemBlockPool> threadContextIt(mThreadContextPool);
			ThreadContext* threadContext = threadContextIt.getNext();

			while (threadContext != NULL)
			{
				threadContext->reset();
				threadContext = threadContextIt.getNext();
			}
		}


		// =========================== Basic methods


		DynamicsLDLContext::DynamicsLDLContext(PxcNpMemBlockPool* memBlockPool,
			PxcScratchAllocator& scratchAllocator,
			Cm::FlushPool& taskPool,
			PxvSimStats& simStats,
			PxTaskManager* taskManager,
			Ps::VirtualAllocatorCallback* allocatorCallback,
			PxsMaterialManager* materialManager,
			IG::IslandSim* accurateIslandSim,
			PxU64 contextID,
			const bool enableStabilization,
			const bool useEnhancedDeterminism,
			const bool useAdaptiveForce,
			const PxReal maxBiasCoefficient,
			const bool frictionEveryIteration
		) :
			Dy::Context(accurateIslandSim, allocatorCallback, simStats, enableStabilization, useEnhancedDeterminism, useAdaptiveForce, maxBiasCoefficient),
			mThreadContextPool(memBlockPool),
			mMaterialManager(materialManager),
			mScratchAllocator(scratchAllocator),
			mTaskPool(taskPool),
			mTaskManager(taskManager),
			mContextID(contextID)
		{
			createThresholdStream(*allocatorCallback);
			createForceChangeThresholdStream(*allocatorCallback);
			mExceededForceThresholdStream[0] = PX_PLACEMENT_NEW(PX_ALLOC(sizeof(ThresholdStream), PX_DEBUG_EXP("ExceededForceThresholdStream[0]")), ThresholdStream(*allocatorCallback));
			mExceededForceThresholdStream[1] = PX_PLACEMENT_NEW(PX_ALLOC(sizeof(ThresholdStream), PX_DEBUG_EXP("ExceededForceThresholdStream[1]")), ThresholdStream(*allocatorCallback));
			mThresholdStreamOut = 0;
			mCurrentIndex = 0;
			mWorldSolverBody.linearVelocity = PxVec3(0);
			mWorldSolverBody.angularState = PxVec3(0);
			mWorldSolverBodyData.invMass = 0;
			mWorldSolverBodyData.sqrtInvInertia = PxMat33(PxZero);
			mWorldSolverBodyData.nodeIndex = IG_INVALID_NODE;
			mWorldSolverBodyData.reportThreshold = PX_MAX_REAL;
			mWorldSolverBodyData.penBiasClamp = -PX_MAX_REAL;
			mWorldSolverBodyData.maxContactImpulse = PX_MAX_REAL;
			mWorldSolverBody.solverProgress = MAX_PERMITTED_SOLVER_PROGRESS;
			mWorldSolverBody.maxSolverNormalProgress = MAX_PERMITTED_SOLVER_PROGRESS;
			mWorldSolverBody.maxSolverFrictionProgress = MAX_PERMITTED_SOLVER_PROGRESS;
			mWorldSolverBodyData.linearVelocity = mWorldSolverBodyData.angularVelocity = PxVec3(0.f);
			mWorldSolverBodyData.body2World = PxTransform(PxIdentity);
			mWorldSolverBodyData.lockFlags = 0;
			//mSolverCore[PxFrictionType::ePATCH] = SolverCoreGeneral::create(frictionEveryIteration);
			mSolverCore[PxFrictionType::ePATCH] = SolverCoreLDL::create(frictionEveryIteration);
			mSolverCore[PxFrictionType::eONE_DIRECTIONAL] = SolverCoreGeneralPF::create();
			mSolverCore[PxFrictionType::eTWO_DIRECTIONAL] = SolverCoreGeneralPF::create();
		}

		DynamicsLDLContext::~DynamicsLDLContext()
		{
			for (PxU32 i = 0; i < PxFrictionType::eFRICTION_COUNT; ++i)
			{
				mSolverCore[i]->destroyV();
			}

			if (mExceededForceThresholdStream[0])
			{
				mExceededForceThresholdStream[0]->~ThresholdStream();
				PX_FREE(mExceededForceThresholdStream[0]);
			}
			mExceededForceThresholdStream[0] = NULL;

			if (mExceededForceThresholdStream[1])
			{
				mExceededForceThresholdStream[1]->~ThresholdStream();
				PX_FREE(mExceededForceThresholdStream[1]);
			}
			mExceededForceThresholdStream[1] = NULL;
		}

#if PX_ENABLE_SIM_STATS
		void DynamicsLDLContext::addThreadStats(const ThreadContext::ThreadSimStats& stats)
		{
			mSimStats.mNbActiveConstraints += stats.numActiveConstraints;
			mSimStats.mNbActiveDynamicBodies += stats.numActiveDynamicBodies;
			mSimStats.mNbActiveKinematicBodies += stats.numActiveKinematicBodies;
			mSimStats.mNbAxisSolverConstraints += stats.numAxisSolverConstraints;
		}
#endif

		// =========================== Solve methods!

		void DynamicsLDLContext::setDescFromIndices(PxSolverConstraintDesc& desc, const PxsIndexedInteraction& constraint, const PxU32 solverBodyOffset)
		{
			PX_COMPILE_TIME_ASSERT(PxsIndexedInteraction::eBODY == 0);
			PX_COMPILE_TIME_ASSERT(PxsIndexedInteraction::eKINEMATIC == 1);
			const PxU32 offsetMap[] = { solverBodyOffset, 0 };
			//const PxU32 offsetMap[] = {mKinematicCount, 0};

			if (constraint.indexType0 == PxsIndexedInteraction::eARTICULATION)
			{
				ArticulationV* a = getArticulation(constraint.articulation0);
				desc.articulationA = a;
				desc.linkIndexA = Ps::to16(getLinkIndex(constraint.articulation0));
			}
			else
			{
				desc.linkIndexA = PxSolverConstraintDesc::NO_LINK;
				//desc.articulationALength = 0; //this is unioned with bodyADataIndex
				/*desc.bodyA = constraint.indexType0 == PxsIndexedInteraction::eWORLD ? &mWorldSolverBody
																					: &mSolverBodyPool[(PxU32)constraint.solverBody0 + offsetMap[constraint.indexType0]];
				desc.bodyADataIndex = PxU16(constraint.indexType0 == PxsIndexedInteraction::eWORLD ? 0
																					: (PxU16)constraint.solverBody0 + 1 + offsetMap[constraint.indexType0]);*/

				desc.bodyA = constraint.indexType0 == PxsIndexedInteraction::eWORLD ? &mWorldSolverBody
					: &mSolverBodyPool[PxU32(constraint.solverBody0) + offsetMap[constraint.indexType0]];
				desc.bodyADataIndex = constraint.indexType0 == PxsIndexedInteraction::eWORLD ? 0
					: PxU32(constraint.solverBody0) + 1 + offsetMap[constraint.indexType0];
			}

			if (constraint.indexType1 == PxsIndexedInteraction::eARTICULATION)
			{
				ArticulationV* b = getArticulation(constraint.articulation1);
				desc.articulationB = b;
				desc.linkIndexB = Ps::to16(getLinkIndex(constraint.articulation1));
			}
			else
			{
				desc.linkIndexB = PxSolverConstraintDesc::NO_LINK;
				//desc.articulationBLength = 0; //this is unioned with bodyBDataIndex
				desc.bodyB = constraint.indexType1 == PxsIndexedInteraction::eWORLD ? &mWorldSolverBody
					: &mSolverBodyPool[PxU32(constraint.solverBody1) + offsetMap[constraint.indexType1]];
				desc.bodyBDataIndex = constraint.indexType1 == PxsIndexedInteraction::eWORLD ? 0
					: PxU32(constraint.solverBody1) + 1 + offsetMap[constraint.indexType1];
			}
		}

		void DynamicsLDLContext::setDescFromIndices(PxSolverConstraintDesc& desc, IG::EdgeIndex edgeIndex, const IG::SimpleIslandManager& islandManager,
			PxU32* bodyRemap, const PxU32 solverBodyOffset)
		{
			PX_COMPILE_TIME_ASSERT(PxsIndexedInteraction::eBODY == 0);
			PX_COMPILE_TIME_ASSERT(PxsIndexedInteraction::eKINEMATIC == 1);

			const IG::IslandSim& islandSim = islandManager.getAccurateIslandSim();

			IG::NodeIndex node1 = islandSim.getNodeIndex1(edgeIndex);
			if (node1.isStaticBody())
			{
				desc.bodyA = &mWorldSolverBody;
				desc.bodyADataIndex = 0;
				desc.linkIndexA = PxSolverConstraintDesc::NO_LINK;
			}
			else
			{
				const IG::Node& node = islandSim.getNode(node1);
				if (node.getNodeType() == IG::Node::eARTICULATION_TYPE)
				{
					Dy::ArticulationV* a = islandSim.getLLArticulation(node1);

					Dy::ArticulationLinkHandle handle;
					PxU8 type;

					a->fillIndexedManager(node1.articulationLinkId(), handle, type);

					if (type == PxsIndexedInteraction::eARTICULATION)
					{
						desc.articulationA = a;
						desc.linkIndexA = Ps::to16(node1.articulationLinkId());
					}
					else
					{
						desc.bodyA = &mWorldSolverBody;
						desc.bodyADataIndex = 0;
						desc.linkIndexA = PxSolverConstraintDesc::NO_LINK;
					}
				}
				else
				{
					PxU32 activeIndex = islandSim.getActiveNodeIndex(node1);
					PxU32 index = node.isKinematic() ? activeIndex : bodyRemap[activeIndex] + solverBodyOffset;
					desc.bodyA = &mSolverBodyPool[index];
					desc.bodyADataIndex = index + 1;
					desc.linkIndexA = PxSolverConstraintDesc::NO_LINK;
				}
			}

			IG::NodeIndex node2 = islandSim.getNodeIndex2(edgeIndex);
			if (node2.isStaticBody())
			{
				desc.bodyB = &mWorldSolverBody;
				desc.bodyBDataIndex = 0;
				desc.linkIndexB = PxSolverConstraintDesc::NO_LINK;
			}
			else
			{
				const IG::Node& node = islandSim.getNode(node2);
				if (node.getNodeType() == IG::Node::eARTICULATION_TYPE)
				{
					Dy::ArticulationV* b = islandSim.getLLArticulation(node2);
					Dy::ArticulationLinkHandle handle;
					PxU8 type;

					b->fillIndexedManager(node2.articulationLinkId(), handle, type);

					if (type == PxsIndexedInteraction::eARTICULATION)
					{
						desc.articulationB = b;
						desc.linkIndexB = Ps::to16(node2.articulationLinkId());
					}
					else
					{
						desc.bodyB = &mWorldSolverBody;
						desc.bodyBDataIndex = 0;
						desc.linkIndexB = PxSolverConstraintDesc::NO_LINK;
					}
				}
				else
				{
					PxU32 activeIndex = islandSim.getActiveNodeIndex(node2);
					PxU32 index = node.isKinematic() ? activeIndex : bodyRemap[activeIndex] + solverBodyOffset;
					desc.bodyB = &mSolverBodyPool[index];
					desc.bodyBDataIndex = index + 1;
					desc.linkIndexB = PxSolverConstraintDesc::NO_LINK;
				}
			}
		}

		
		void DynamicsLDLContext::update(IG::SimpleIslandManager& simpleIslandManager, PxBaseTask* continuation, PxBaseTask* lostTouchTask,
			PxsContactManager** /*foundPatchManagers*/, PxU32 /*nbFoundPatchManagers*/,
			PxsContactManager** /*lostPatchManagers*/, PxU32 /*nbLostPatchManagers*/,
			PxU32 /*maxPatchesPerCM*/,
			PxsContactManagerOutputIterator& iterator,
			PxsContactManagerOutput*,
			const PxReal dt, const PxVec3& gravity, const PxU32 /*bitMapWordCounts*/)
		{
			PX_PROFILE_ZONE("Dynamics.solverQueueTasks", mContextID);

			PX_UNUSED(simpleIslandManager);

			mOutputIterator = iterator;

			mDt = dt;
			mInvDt = dt == 0.0f ? 0.0f : 1.0f / dt;
			mGravity = gravity;

			const IG::IslandSim& islandSim = simpleIslandManager.getAccurateIslandSim();

			const PxU32 islandCount = islandSim.getNbActiveIslands();

			const PxU32 activatedContactCount = islandSim.getNbActivatedEdges(IG::Edge::eCONTACT_MANAGER);
			const IG::EdgeIndex* const activatingEdges = islandSim.getActivatedEdges(IG::Edge::eCONTACT_MANAGER);

			for (PxU32 a = 0; a < activatedContactCount; ++a)
			{
				PxsContactManager* cm = simpleIslandManager.getContactManager(activatingEdges[a]);
				if (cm)
				{
					cm->getWorkUnit().frictionPatchCount = 0; //KS - zero the friction patch count on any activating edges
				}
			}

#if PX_ENABLE_SIM_STATS
			if (islandCount > 0)
			{
				mSimStats.mNbActiveKinematicBodies = islandSim.getNbActiveKinematics();
				mSimStats.mNbActiveDynamicBodies = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);
				mSimStats.mNbActiveConstraints = islandSim.getNbActiveEdges(IG::Edge::eCONSTRAINT);
			}
			else
			{
				mSimStats.mNbActiveKinematicBodies = islandSim.getNbActiveKinematics();
				mSimStats.mNbActiveDynamicBodies = 0;
				mSimStats.mNbActiveConstraints = 0;
			}
#endif

			mThresholdStreamOut = 0;

			resetThreadContexts();

			//If there is no work to do then we can do nothing at all.
			if (0 == islandCount)
			{
				return;
			}

			//Block to make sure it doesn't run before stage2 of update!
			lostTouchTask->addReference();

			UpdateContinuationTask* task = PX_PLACEMENT_NEW(mTaskPool.allocate(sizeof(UpdateContinuationTask)), UpdateContinuationTask)
				(*this, simpleIslandManager, lostTouchTask, mContextID);

			task->setContinuation(continuation);

			//KS - test that world solver body's velocities are finite and 0, then set it to 0.
			//Technically, the velocity should always be 0 but can be stomped if a NAN creeps into the simulation.
			PX_ASSERT(mWorldSolverBody.linearVelocity == PxVec3(0.f));
			PX_ASSERT(mWorldSolverBody.angularState == PxVec3(0.f));
			PX_ASSERT(mWorldSolverBody.linearVelocity.isFinite());
			PX_ASSERT(mWorldSolverBody.angularState.isFinite());

			mWorldSolverBody.linearVelocity = mWorldSolverBody.angularState = PxVec3(0.f);

			const PxU32 kinematicCount = islandSim.getNbActiveKinematics();
			const IG::NodeIndex* const kinematicIndices = islandSim.getActiveKinematics();
			mKinematicCount = kinematicCount;

			const PxU32 bodyCount = islandSim.getNbActiveNodes(IG::Node::eRIGID_BODY_TYPE);

			PxU32 numArtics = islandSim.getNbActiveNodes(IG::Node::eARTICULATION_TYPE);

			{
				if (kinematicCount + bodyCount > mSolverBodyPool.capacity())
				{
					mSolverBodyPool.reserve((kinematicCount + bodyCount + 31) & ~31); // pad out to 32 * 128 = 4k to prevent alloc churn
					mSolverBodyDataPool.reserve((kinematicCount + bodyCount + 31 + 1) & ~31); // pad out to 32 * 128 = 4k to prevent alloc churn
					mSolverBodyRemapTable.reserve((kinematicCount + bodyCount + 31 + 1) & ~31);
				}

				{
					PxSolverBody emptySolverBody;
					PxMemZero(&emptySolverBody, sizeof(PxSolverBody));
					mSolverBodyPool.resize(kinematicCount + bodyCount, emptySolverBody);
					PxSolverBodyData emptySolverBodyData;
					PxMemZero(&emptySolverBodyData, sizeof(PxSolverBodyData));
					mSolverBodyDataPool.resize(kinematicCount + bodyCount + 1, emptySolverBodyData);
					mSolverBodyRemapTable.resize(bodyCount);
				}

				// integrate and copy all the kinematics - overkill, since not all kinematics
				// need solver bodies

				mSolverBodyDataPool[0] = mWorldSolverBodyData;

				{
					PX_PROFILE_ZONE("Dynamics.updateKinematics", mContextID);
					PxMemZero(mSolverBodyPool.begin(), kinematicCount * sizeof(PxSolverBody));
					for (PxU32 i = 0; i < kinematicCount; i += KinematicCopyTask::NbKinematicsPerTask)
					{
						const PxU32 nbToProcess = PxMin(KinematicCopyTask::NbKinematicsPerTask, kinematicCount - i);

						KinematicCopyTask* copyTask = PX_PLACEMENT_NEW(mTaskPool.allocate(sizeof(KinematicCopyTask)), KinematicCopyTask)
							(&kinematicIndices[i], nbToProcess, islandSim, &mSolverBodyDataPool[i], mContextID);

						copyTask->setContinuation(task);

						copyTask->removeReference();
					}
				}
			}

			//Resize arrays of solver constraints...

			PxU32 numArticulationConstraints = numArtics * Dy::DY_ARTICULATION_MAX_SIZE; //Just allocate enough memory to fit worst-case maximum size articulations...

			const PxU32 nbActiveContactManagers = islandSim.getNbActiveEdges(IG::Edge::eCONTACT_MANAGER);
			const PxU32 nbActiveConstraints = islandSim.getNbActiveEdges(IG::Edge::eCONSTRAINT);

			PxU32 totalConstraintCount = nbActiveConstraints + nbActiveContactManagers + numArticulationConstraints;

			mSolverConstraintDescPool.forceSize_Unsafe(0);
			mSolverConstraintDescPool.reserve((totalConstraintCount + 63) & (~63));
			mSolverConstraintDescPool.forceSize_Unsafe(totalConstraintCount);

			mOrderedSolverConstraintDescPool.forceSize_Unsafe(0);
			mOrderedSolverConstraintDescPool.reserve((totalConstraintCount + 63) & (~63));
			mOrderedSolverConstraintDescPool.forceSize_Unsafe(totalConstraintCount);

			mTempSolverConstraintDescPool.forceSize_Unsafe(0);
			mTempSolverConstraintDescPool.reserve((totalConstraintCount + 63) & (~63));
			mTempSolverConstraintDescPool.forceSize_Unsafe(totalConstraintCount);

			mContactConstraintBatchHeaders.forceSize_Unsafe(0);
			mContactConstraintBatchHeaders.reserve((totalConstraintCount + 63) & (~63));
			mContactConstraintBatchHeaders.forceSize_Unsafe(totalConstraintCount);

			mContactList.forceSize_Unsafe(0);
			mContactList.reserve((nbActiveContactManagers + 63u) & (~63u));
			mContactList.forceSize_Unsafe(nbActiveContactManagers);

			mMotionVelocityArray.forceSize_Unsafe(0);
			mMotionVelocityArray.reserve((bodyCount + 63u) & (~63u));
			mMotionVelocityArray.forceSize_Unsafe(bodyCount);

			mBodyCoreArray.forceSize_Unsafe(0);
			mBodyCoreArray.reserve((bodyCount + 63u) & (~63u));
			mBodyCoreArray.forceSize_Unsafe(bodyCount);

			mRigidBodyArray.forceSize_Unsafe(0);
			mRigidBodyArray.reserve((bodyCount + 63u) & (~63u));
			mRigidBodyArray.forceSize_Unsafe(bodyCount);

			mArticulationArray.forceSize_Unsafe(0);
			mArticulationArray.reserve((numArtics + 63u) & (~63u));
			mArticulationArray.forceSize_Unsafe(numArtics);

			mNodeIndexArray.forceSize_Unsafe(0);
			mNodeIndexArray.reserve((bodyCount + 63u) & (~63u));
			mNodeIndexArray.forceSize_Unsafe(bodyCount);

			ThresholdStream& stream = getThresholdStream();
			stream.forceSize_Unsafe(0);
			stream.reserve(Ps::nextPowerOfTwo(nbActiveContactManagers != 0 ? nbActiveContactManagers - 1 : nbActiveContactManagers));

			//flip exceeded force threshold buffer
			mCurrentIndex = 1 - mCurrentIndex;

			task->removeReference();
		}

		
		PxBaseTask* createSolverTaskChain(DynamicsLDLContext& dynamicContext,
			const SolverIslandObjects& objects,
			const PxsIslandIndices& counts,
			const PxU32 solverBodyOffset,
			IG::SimpleIslandManager& islandManager,
			PxU32* bodyRemapTable, PxsMaterialManager* materialManager, PxBaseTask* continuation,
			PxsContactManagerOutputIterator& iterator, bool useEnhancedDeterminism)
		{
			Cm::FlushPool& taskPool = dynamicContext.getTaskPool();

			taskPool.lock();

			IslandContext* islandContext = reinterpret_cast<IslandContext*>(taskPool.allocate(sizeof(IslandContext)));
			islandContext->mThreadContext = NULL;
			islandContext->mCounts = counts;

			// create lead task
			PxsSolverStartTask* startTask = PX_PLACEMENT_NEW(taskPool.allocateNotThreadSafe(sizeof(PxsSolverStartTask)), PxsSolverStartTask)(dynamicContext, *islandContext, objects, solverBodyOffset, dynamicContext.getKinematicCount(),
				islandManager, bodyRemapTable, materialManager, iterator, useEnhancedDeterminism);
			PxsSolverEndTask* endTask = PX_PLACEMENT_NEW(taskPool.allocateNotThreadSafe(sizeof(PxsSolverEndTask)), PxsSolverEndTask)(dynamicContext, *islandContext, objects, solverBodyOffset, iterator);

			PxsSolverCreateFinalizeConstraintsTask* createFinalizeConstraintsTask = PX_PLACEMENT_NEW(taskPool.allocateNotThreadSafe(sizeof(PxsSolverCreateFinalizeConstraintsTask)), PxsSolverCreateFinalizeConstraintsTask)(dynamicContext, *islandContext, solverBodyOffset, iterator, useEnhancedDeterminism);
			PxsSolverSetupSolveTask* setupSolveTask = PX_PLACEMENT_NEW(taskPool.allocateNotThreadSafe(sizeof(PxsSolverSetupSolveTask)), PxsSolverSetupSolveTask)(dynamicContext, *islandContext, objects, solverBodyOffset, islandManager.getAccurateIslandSim());

			PxsSolverConstraintPartitionTask* partitionConstraintsTask = PX_PLACEMENT_NEW(taskPool.allocateNotThreadSafe(sizeof(PxsSolverConstraintPartitionTask)), PxsSolverConstraintPartitionTask)(dynamicContext, *islandContext, objects, solverBodyOffset, useEnhancedDeterminism);

			endTask->setContinuation(continuation);

			// set up task chain in reverse order
			chainTasksLDL(setupSolveTask, endTask);
			chainTasksLDL(createFinalizeConstraintsTask, setupSolveTask);
			chainTasksLDL(partitionConstraintsTask, createFinalizeConstraintsTask);
			chainTasksLDL(startTask, partitionConstraintsTask);

			taskPool.unlock();

			return startTask;
		}

		void DynamicsLDLContext::updatePostKinematic(IG::SimpleIslandManager& simpleIslandManager, PxBaseTask* /*continuation*/, PxBaseTask* lostTouchTask)
		{
			const IG::IslandSim& islandSim = simpleIslandManager.getAccurateIslandSim();

			const PxU32 islandCount = islandSim.getNbActiveIslands();

			PxU32 constraintIndex = 0;

			PxU32 solverBatchMax = mSolverBatchSize;
			PxU32 articulationBatchMax = mSolverArticBatchSize;
			PxU32 minimumConstraintCount = 1;

			//create force threshold tasks to produce force change events
			PxsForceThresholdTask* forceThresholdTask = PX_PLACEMENT_NEW(getTaskPool().allocate(sizeof(PxsForceThresholdTask)), PxsForceThresholdTask)(*this);
			forceThresholdTask->setContinuation(lostTouchTask);

			const IG::IslandId* const islandIds = islandSim.getActiveIslands();

			PxU32 currentIsland = 0;
			PxU32 currentBodyIndex = 0;
			PxU32 currentArticulation = 0;
			PxU32 currentContact = 0;
			//while(start<sentinel)
			while (currentIsland < islandCount)
			{
				SolverIslandObjects objectStarts;
				objectStarts.articulations = mArticulationArray.begin() + currentArticulation;
				objectStarts.bodies = mRigidBodyArray.begin() + currentBodyIndex;
				objectStarts.contactManagers = mContactList.begin() + currentContact;
				objectStarts.constraintDescs = mSolverConstraintDescPool.begin() + constraintIndex;
				objectStarts.orderedConstraintDescs = mOrderedSolverConstraintDescPool.begin() + constraintIndex;
				objectStarts.tempConstraintDescs = mTempSolverConstraintDescPool.begin() + constraintIndex;
				objectStarts.constraintBatchHeaders = mContactConstraintBatchHeaders.begin() + constraintIndex;
				objectStarts.motionVelocities = mMotionVelocityArray.begin() + currentBodyIndex;
				objectStarts.bodyCoreArray = mBodyCoreArray.begin() + currentBodyIndex;
				objectStarts.islandIds = islandIds + currentIsland;
				objectStarts.bodyRemapTable = mSolverBodyRemapTable.begin();
				objectStarts.nodeIndexArray = mNodeIndexArray.begin() + currentBodyIndex;

				PxU32 startIsland = currentIsland;
				PxU32 constraintCount = 0;

				PxU32 nbArticulations = 0;
				PxU32 nbBodies = 0;
				PxU32 nbConstraints = 0;
				PxU32 nbContactManagers = 0;

				//KS - logic is a bit funky here. We will keep rolling the island together provided currentIsland < islandCount AND either we haven't exceeded the max number of bodies or we have
				//zero constraints AND we haven't exceeded articulation batch counts (it's still currently beneficial to keep articulations in separate islands but this is only temporary).
				while ((currentIsland < islandCount && (nbBodies < solverBatchMax || constraintCount < minimumConstraintCount)) && nbArticulations < articulationBatchMax)
				{
					const IG::Island& island = islandSim.getIsland(islandIds[currentIsland]);
					nbBodies += island.mSize[IG::Node::eRIGID_BODY_TYPE];
					nbArticulations += island.mSize[IG::Node::eARTICULATION_TYPE];
					nbConstraints += island.mEdgeCount[IG::Edge::eCONSTRAINT];
					nbContactManagers += island.mEdgeCount[IG::Edge::eCONTACT_MANAGER];
					constraintCount = nbConstraints + nbContactManagers;
					currentIsland++;
				}

				objectStarts.numIslands = currentIsland - startIsland;

				constraintIndex += nbArticulations * Dy::DY_ARTICULATION_MAX_SIZE;

				PxsIslandIndices counts;

				counts.articulations = nbArticulations;
				counts.bodies = nbBodies;

				counts.constraints = nbConstraints;
				counts.contactManagers = nbContactManagers;
				if (counts.articulations + counts.bodies > 0)
				{
					PxBaseTask* task = createSolverTaskChain(*this, objectStarts, counts,
						mKinematicCount + currentBodyIndex, simpleIslandManager, mSolverBodyRemapTable.begin(), mMaterialManager, forceThresholdTask, mOutputIterator, mUseEnhancedDeterminism);
					task->removeReference();
				}

				currentBodyIndex += nbBodies;
				currentArticulation += nbArticulations;
				currentContact += nbContactManagers;

				constraintIndex += constraintCount;
			}

			//kick off forceThresholdTask
			forceThresholdTask->removeReference();
		}

		void DynamicsLDLContext::updateBodyCore(PxBaseTask* continuation)
		{
			PX_UNUSED(continuation);
		}

		void DynamicsLDLContext::mergeResults()
		{
			PX_PROFILE_ZONE("Dynamics.solverMergeResults", mContextID);
			//OK. Sum up sim stats here...

#if PX_ENABLE_SIM_STATS
			PxcThreadCoherentCacheIterator<ThreadContext, PxcNpMemBlockPool> threadContextIt(mThreadContextPool);
			ThreadContext* threadContext = threadContextIt.getNext();

			while (threadContext != NULL)
			{
				ThreadContext::ThreadSimStats& threadStats = threadContext->getSimStats();
				addThreadStats(threadStats);
				threadStats.clear();
				threadContext = threadContextIt.getNext();
			}
#endif
		}

		void DynamicsLDLContext::preIntegrationParallel(
			const PxF32 dt,
			PxsBodyCore* const* bodyArray,					// INOUT: core body attributes
			PxsRigidBody* const* originalBodyArray,			// IN: original bodies (LEGACY - DON'T deref the ptrs!!)
			PxU32 const* nodeIndexArray,						// IN: island node index
			PxU32 bodyCount,									// IN: body count
			PxSolverBody* solverBodyPool,					// IN: solver body pool (space preallocated)
			PxSolverBodyData* solverBodyDataPool,			// IN: solver body data pool (space preallocated)
			Cm::SpatialVector* /*motionVelocityArray*/,			// OUT: motion velocities
			PxU32& maxSolverPositionIterations,
			PxU32& maxSolverVelocityIterations,
			PxBaseTask& task
		)
		{
			//TODO - make this based on some variables so we can try different configurations
			const PxU32 IntegrationPerThread = 256;

			const PxU32 numTasks = ((bodyCount + IntegrationPerThread - 1) / IntegrationPerThread);
			const PxU32 taskBatchSize = 64;

			for (PxU32 i = 0; i < numTasks; i += taskBatchSize)
			{
				const PxU32 nbTasks = PxMin(numTasks - i, taskBatchSize);
				PxsPreIntegrateTask* tasks = reinterpret_cast<PxsPreIntegrateTask*>(getTaskPool().allocate(sizeof(PxsPreIntegrateTask) * nbTasks));
				for (PxU32 a = 0; a < nbTasks; ++a)
				{
					PxU32 startIndex = (i + a) * IntegrationPerThread;
					PxU32 nbToIntegrate = PxMin((bodyCount - startIndex), IntegrationPerThread);
					PxsPreIntegrateTask* pTask = PX_PLACEMENT_NEW(&tasks[a], PxsPreIntegrateTask)(*this, bodyArray,
						originalBodyArray, nodeIndexArray, solverBodyPool, solverBodyDataPool, dt, bodyCount,
						&maxSolverPositionIterations, &maxSolverVelocityIterations, startIndex,
						nbToIntegrate, mGravity);

					pTask->setContinuation(&task);
					pTask->removeReference();
				}
			}

			PxMemZero(solverBodyPool, bodyCount * sizeof(PxSolverBody));
		}

		void DynamicsLDLContext::solveParallel(SolverIslandParams& params, IG::IslandSim& islandSim, Cm::SpatialVectorF* Z, Cm::SpatialVectorF* deltaV)
		{
			PxI32 targetCount = mSolverCore[mFrictionType]->solveVParallelAndWriteBack(params, Z, deltaV);

			PxI32* solveCount = &params.constraintIndex2;

			//PxI32 targetCount = (PxI32)(params.numConstraintHeaders * (params.velocityIterations + params.positionIterations));

			WAIT_FOR_PROGRESS_NO_TIMER(solveCount, targetCount);

			integrateCoreParallel(params, islandSim);
		}

		void DynamicsLDLContext::integrateCoreParallel(SolverIslandParams& params, IG::IslandSim& islandSim)
		{
			const PxI32 unrollCount = 128;

			PxI32* bodyIntegrationListIndex = &params.bodyIntegrationListIndex;

			PxI32 index = physx::shdfnd::atomicAdd(bodyIntegrationListIndex, unrollCount) - unrollCount;

			const PxI32 numBodies = PxI32(params.bodyListSize);
			const PxI32 numArtics = PxI32(params.articulationListSize);

			Cm::SpatialVector* PX_RESTRICT motionVelocityArray = params.motionVelocityArray;
			PxsBodyCore* const* bodyArray = params.bodyArray;
			PxsRigidBody** PX_RESTRICT rigidBodies = params.rigidBodies;
			ArticulationSolverDesc* PX_RESTRICT articulationListStart = params.articulationListStart;

			PxI32 numIntegrated = 0;

			PxI32 bodyRemainder = unrollCount;

			while (index < numArtics)
			{
				const PxI32 remainder = PxMin(numArtics - index, unrollCount);
				bodyRemainder -= remainder;

				for (PxI32 a = 0; a < remainder; ++a, index++)
				{
					const PxI32 i = index;
					{
						//PX_PROFILE_ZONE("Articulations.integrate", mContextID);

						ArticulationPImpl::updateBodies(articulationListStart[i], mDt);
					}

					++numIntegrated;
				}
				if (bodyRemainder == 0)
				{
					index = physx::shdfnd::atomicAdd(bodyIntegrationListIndex, unrollCount) - unrollCount;
					bodyRemainder = unrollCount;
				}
			}

			index -= numArtics;

			const PxI32 unrollPlusArtics = unrollCount + numArtics;

			PxSolverBody* PX_RESTRICT solverBodies = params.bodyListStart;
			PxSolverBodyData* PX_RESTRICT solverBodyData = params.bodyDataList + params.solverBodyOffset + 1;

			while (index < numBodies)
			{
				const PxI32 remainder = PxMin(numBodies - index, bodyRemainder);
				bodyRemainder -= remainder;
				for (PxI32 a = 0; a < remainder; ++a, index++)
				{
					const PxI32 prefetch = PxMin(index + 4, numBodies - 1);
					Ps::prefetchLine(bodyArray[prefetch]);
					Ps::prefetchLine(bodyArray[prefetch], 128);
					Ps::prefetchLine(&solverBodies[index], 128);
					Ps::prefetchLine(&motionVelocityArray[index], 128);
					Ps::prefetchLine(&bodyArray[index + 32]);
					Ps::prefetchLine(&rigidBodies[prefetch]);

					PxSolverBodyData& data = solverBodyData[index];

					integrateCore(motionVelocityArray[index].linear, motionVelocityArray[index].angular,
						solverBodies[index], data, mDt);

					PxsRigidBody& rBody = *rigidBodies[index];
					PxsBodyCore& core = rBody.getCore();
					rBody.mLastTransform = core.body2World;
					core.body2World = data.body2World;
					core.linearVelocity = data.linearVelocity;
					core.angularVelocity = data.angularVelocity;

					bool hasStaticTouch = islandSim.getIslandStaticTouchCount(IG::NodeIndex(data.nodeIndex)) != 0;
					sleepCheck(rigidBodies[index], mDt, mInvDt, mEnableStabilization, mUseAdaptiveForce, motionVelocityArray[index], hasStaticTouch);

					++numIntegrated;
				}

				{
					index = physx::shdfnd::atomicAdd(bodyIntegrationListIndex, unrollCount) - unrollPlusArtics;
					bodyRemainder = unrollCount;
				}
			}

			Ps::memoryBarrier();
			physx::shdfnd::atomicAdd(&params.numObjectsIntegrated, numIntegrated);
		}
	}
}