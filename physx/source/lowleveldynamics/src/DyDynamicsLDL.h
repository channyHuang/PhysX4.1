#ifndef DY_DYNAMICS_LDL_H
#define DY_DYNAMICS_LDL_H

#include "PxPhysicsAPI.h"

#include "PxvConfig.h"
#include "CmSpatialVector.h"
#include "CmTask.h"
#include "CmPool.h"
#include "PxcThreadCoherentCache.h"
#include "DyThreadContext.h"
#include "PxcConstraintBlockStream.h"
#include "DySolverBody.h"
#include "DyContext.h"
#include "PxsIslandManagerTypes.h"
#include "PxvNphaseImplementationContext.h"
#include "solver/PxSolverDefs.h"

#include "DyDynamics.h"

namespace physx
{
	namespace Dy {
		class SolverCore;
		struct SolverIslandParams;
		struct ArticulationSolverDesc;
		class Articulation;
		class DynamicsLDLContext;

#define SOLVER_PARALLEL_METHOD_ARGS_LDL									\
	DynamicsLDLContext&	context,									\
	SolverIslandParams& params,										\
	IG::IslandSim& islandSim

		void solveParallel(SOLVER_PARALLEL_METHOD_ARGS_LDL);
		void solveParallelCouloumFriction(SOLVER_PARALLEL_METHOD_ARGS_LDL);

		class DynamicsLDLContext : public Context
		{
			PX_NOCOPY(DynamicsLDLContext)
		public:

			/**
			\brief Creates a DynamicsContext associated with a PxsContext
			\return A pointer to the newly-created DynamicsContext.
			*/
			static DynamicsLDLContext* create(PxcNpMemBlockPool* memBlockPool,
				PxcScratchAllocator& scratchAllocator,
				Cm::FlushPool& taskPool,
				PxvSimStats& simStats,
				PxTaskManager* taskManager,
				Ps::VirtualAllocatorCallback* allocator,
				PxsMaterialManager* materialManager,
				IG::IslandSim* accurateIslandSim,
				PxU64 contextID,
				const bool enableStabilization,
				const bool useEnhancedDeterminism,
				const bool useAdaptiveForce,
				const PxReal maxBiasCoefficient,
				const bool frictionEveryIteration
			);

			/**
			\brief Destroys this DynamicsContext
			*/
			void						destroy();

			/**
			\brief Returns the static world solver body
			\return The static world solver body.
			*/
			PX_FORCE_INLINE PxSolverBody& getWorldSolverBody() { return mWorldSolverBody; }

			PX_FORCE_INLINE Cm::FlushPool& getTaskPool() { return mTaskPool; }

			PX_FORCE_INLINE ThresholdStream& getThresholdStream() { return *mThresholdStream; }

			PX_FORCE_INLINE PxvSimStats& getSimStats() { return mSimStats; }

#if PX_ENABLE_SIM_STATS
			void									addThreadStats(const ThreadContext::ThreadSimStats& stats);
#endif

			/**
			\brief The entry point for the constraint solver.
			\param[in]	dt	The simulation time-step
			\param[in]	continuation The continuation task for the solver

			This method is called after the island generation has completed. Its main responsibilities are:
			(1) Reserving the solver body pools
			(2) Initializing the static and kinematic solver bodies, which are shared resources between islands.
			(3) Construct the solver task chains for each island

			Each island is solved as an independent solver task chain in parallel.

			*/

			virtual void						update(IG::SimpleIslandManager& simpleIslandManager, PxBaseTask* continuation, PxBaseTask* lostTouchTask,
				PxsContactManager** foundPatchManagers, PxU32 nbFoundPatchManagers, PxsContactManager** lostPatchManagers, PxU32 nbLostPatchManagers,
				PxU32 maxPatchesPerCM, PxsContactManagerOutputIterator& iter, PxsContactManagerOutput* gpuOutputs, const PxReal dt, const PxVec3& gravity, const PxU32 bitMapWordCounts);


			void updatePostKinematic(IG::SimpleIslandManager& simpleIslandManager, PxBaseTask* continuation, PxBaseTask* lostTouchTask);

			virtual void						processLostPatches(IG::SimpleIslandManager& /*simpleIslandManager*/, PxsContactManager** /*lostPatchManagers*/, PxU32 /*nbLostPatchManagers*/, PxsContactManagerOutputIterator& /*iterator*/) {}

			virtual void						updateBodyCore(PxBaseTask* continuation);

			virtual void						setSimulationController(PxsSimulationController* simulationController) { mSimulationController = simulationController; }
			/**
			\brief This method combines the results of several islands, e.g. constructing scene-level simulation statistics and merging together threshold streams for contact notification.
			*/
			virtual void							mergeResults();

			virtual void							getDataStreamBase(void*& /*contactStreamBase*/, void*& /*patchStreamBase*/, void*& /*forceAndIndicesStreamBase*/) {}

			/**
			\brief Allocates and returns a thread context object.
			\return A thread context.
			*/
			PX_FORCE_INLINE ThreadContext* getThreadContext()
			{
				return mThreadContextPool.get();
			}

			/**
			\brief Returns a thread context to the thread context pool.
			\param[in] context The thread context to return to the thread context pool.
			*/
			void								putThreadContext(ThreadContext* context)
			{
				mThreadContextPool.put(context);
			}


			PX_FORCE_INLINE	PxU32					getKinematicCount()		const { return mKinematicCount; }
			PX_FORCE_INLINE	PxU64					getContextId()			const { return mContextID; }

		protected:

			/**
			\brief Constructor for DynamicsContext
			*/
			DynamicsLDLContext(PxcNpMemBlockPool* memBlockPool,
				PxcScratchAllocator& scratchAllocator,
				Cm::FlushPool& taskPool,
				PxvSimStats& simStats,
				PxTaskManager* taskManager,
				Ps::VirtualAllocatorCallback* allocator,
				PxsMaterialManager* materialManager,
				IG::IslandSim* accurateIslandSim,
				PxU64 contextID,
				const bool enableStabilization,
				const bool useEnhancedDeterminism,
				const bool useAdaptiveForce,
				const PxReal maxBiasCoefficient,
				const bool frictionEveryIteration
			);
			/**
			\brief Destructor for DynamicsContext
			*/
			virtual								~DynamicsLDLContext();


			// Solver helper-methods
			/**
			\brief Computes the unconstrained velocity for a given PxsRigidBody
			\param[in] atom The PxsRigidBody
			*/
			void								computeUnconstrainedVelocity(PxsRigidBody* atom)	const;

			/**
			\brief fills in a PxSolverConstraintDesc from an indexed interaction
			\param[in,out] desc The PxSolverConstraintDesc
			\param[in] constraint The PxsIndexedInteraction
			*/
			void								setDescFromIndices(PxSolverConstraintDesc& desc,
				const PxsIndexedInteraction& constraint, const PxU32 solverBodyOffset);


			void								setDescFromIndices(PxSolverConstraintDesc& desc, IG::EdgeIndex edgeIndex,
				const IG::SimpleIslandManager& islandManager, PxU32* bodyRemapTable, const PxU32 solverBodyOffset);

			/**
			\brief Compute the unconstrained velocity for set of bodies in parallel. This function may spawn additional tasks.
			\param[in] dt The timestep
			\param[in] bodyArray The array of body cores
			\param[in] originalBodyArray The array of PxsRigidBody
			\param[in] nodeIndexArray The array of island node index
			\param[in] bodyCount The number of bodies
			\param[out] solverBodyPool The pool of solver bodies. These are synced with the corresponding body in bodyArray.
			\param[out] solverBodyDataPool The pool of solver body data. These are synced with the corresponding body in bodyArray
			\param[out] motionVelocityArray The motion velocities for the bodies
			\param[out] maxSolverPositionIterations The maximum number of position iterations requested by any body in the island
			\param[out] maxSolverVelocityIterations The maximum number of velocity iterations requested by any body in the island
			\param[out] integrateTask The continuation task for any tasks spawned by this function.
			*/
			void								preIntegrationParallel(
				const PxF32 dt,
				PxsBodyCore* const* bodyArray,					// INOUT: core body attributes
				PxsRigidBody* const* originalBodyArray,			// IN: original body atom names (LEGACY - DON'T deref the ptrs!!)
				PxU32 const* nodeIndexArray,						// IN: island node index
				PxU32 bodyCount,									// IN: body count
				PxSolverBody* solverBodyPool,					// IN: solver atom pool (space preallocated)
				PxSolverBodyData* solverBodyDataPool,
				Cm::SpatialVector* motionVelocityArray,			// OUT: motion velocities
				PxU32& maxSolverPositionIterations,
				PxU32& maxSolverVelocityIterations,
				PxBaseTask& integrateTask
			);

			/**
			\brief Solves an island in parallel.

			\param[in] params Solver parameter structure
			*/

			void								solveParallel(SolverIslandParams& params, IG::IslandSim& islandSim, Cm::SpatialVectorF* Z, Cm::SpatialVectorF* deltaV);



			void								integrateCoreParallel(SolverIslandParams& params, IG::IslandSim& islandSim);




			/**
			\brief Resets the thread contexts
			*/
			void									resetThreadContexts();

			/**
			\brief Returns the scratch memory allocator.
			\return The scratch memory allocator.
			*/
			PX_FORCE_INLINE PxcScratchAllocator& getScratchAllocator() { return mScratchAllocator; }

			//Data

			/**
			\brief Body to represent the world static body.
			*/
			PX_ALIGN(16, PxSolverBody				mWorldSolverBody);
			/**
			\brief Body data to represent the world static body.
			*/
			PX_ALIGN(16, PxSolverBodyData			mWorldSolverBodyData);

			/**
			\brief A thread context pool
			*/
			PxcThreadCoherentCache<ThreadContext, PxcNpMemBlockPool> mThreadContextPool;

			/**
			\brief Solver constraint desc array
			*/
			SolverConstraintDescPool	mSolverConstraintDescPool;

			/**
			\brief Ordered sover constraint desc array (after partitioning)
			*/
			SolverConstraintDescPool	mOrderedSolverConstraintDescPool;

			/**
			\brief A temporary array of constraint descs used for partitioning
			*/
			SolverConstraintDescPool	mTempSolverConstraintDescPool;

			/**
			\brief An array of contact constraint batch headers
			*/
			Ps::Array<PxConstraintBatchHeader> mContactConstraintBatchHeaders;

			/**
			\brief Array of motion velocities for all bodies in the scene.
			*/
			Ps::Array<Cm::SpatialVector> mMotionVelocityArray;

			/**
			\brief Array of body core pointers for all bodies in the scene.
			*/
			Ps::Array<PxsBodyCore*>	mBodyCoreArray;

			/**
			\brief Array of rigid body pointers for all bodies in the scene.
			*/
			Ps::Array<PxsRigidBody*> mRigidBodyArray;

			/**
			\brief Array of articulationpointers for all articulations in the scene.
			*/
			Ps::Array<ArticulationV*> mArticulationArray;

			/**
			\brief Global pool for solver bodies. Kinematic bodies are at the start, and then dynamic bodies
			*/
			SolverBodyPool			mSolverBodyPool;
			/**
			\brief Global pool for solver body data. Kinematic bodies are at the start, and then dynamic bodies
			*/
			SolverBodyDataPool		mSolverBodyDataPool;


			ThresholdStream* mExceededForceThresholdStream[2]; //this store previous and current exceeded force thresholdStream	

			Ps::Array<PxU32>		mExceededForceThresholdStreamMask;

			/**
			\brief Interface to the solver core.
			\note We currently only support PxsSolverCoreSIMD. Other cores may be added in future releases.
			*/
			SolverCore* mSolverCore[PxFrictionType::eFRICTION_COUNT];

			Ps::Array<PxU32>		mSolverBodyRemapTable;				//Remaps from the "active island" index to the index within a solver island

			Ps::Array<PxU32>		mNodeIndexArray;					//island node index

			Ps::Array<PxsIndexedContactManager> mContactList;

			/**
			\brief The total number of kinematic bodies in the scene
			*/
			PxU32						mKinematicCount;

			/**
			\brief Atomic counter for the number of threshold stream elements.
			*/
			PxI32						mThresholdStreamOut;



			PxsMaterialManager* mMaterialManager;

			PxsContactManagerOutputIterator mOutputIterator;

		private:
			//private:
			PxcScratchAllocator& mScratchAllocator;
			Cm::FlushPool& mTaskPool;
			PxTaskManager* mTaskManager;
			PxU32										mCurrentIndex; // this is the index point to the current exceeded force threshold stream

			PxU64										mContextID;

		protected:

			friend class PxsSolverStartTask;
			friend class PxsSolverAticulationsTask;
			friend class PxsSolverSetupConstraintsTask;
			friend class PxsSolverCreateFinalizeConstraintsTask;
			friend class PxsSolverConstraintPartitionTask;
			friend class PxsSolverSetupSolveTask;
			friend class PxsSolverIntegrateTask;
			friend class PxsSolverEndTask;
			friend class PxsSolverConstraintPostProcessTask;
			friend class PxsForceThresholdTask;
			friend class SolverArticulationUpdateTask;

			friend void solveParallel(SOLVER_PARALLEL_METHOD_ARGS_LDL);
		};
	}
}

#endif //DY_DYNAMICS_H
