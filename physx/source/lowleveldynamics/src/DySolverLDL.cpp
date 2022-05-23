#include "PsAllocator.h"
#include <new>
#include <stdio.h>
#include "CmPhysXCommon.h"
#include "DySolverBody.h"
#include "DySolverConstraint1D.h"
#include "DySolverContact.h"
#include "DyThresholdTable.h"
#include "DySolverControl.h"
#include "DySolverLDL.h"
#include "DyArticulationHelper.h"
#include "PsAtomic.h"
#include "PsIntrinsics.h"
#include "DyArticulationPImpl.h"
#include "PsThread.h"
#include "DySolverConstraintDesc.h"
#include "DySolverContext.h"

#include "DyMathLDL.h"

#include <vector>

namespace physx
{
	namespace Dy
	{
		void solve1DBlock(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveContactBlock(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveExtContactBlock(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveExt1DBlock(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveContact_BStaticBlock(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveContactPreBlock(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveContactPreBlock_Static(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solve1D4_Block(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);

		void solve1DBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveContactBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache);
		void solveExtContactBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(params);
			solveExtContactBlock(desc, constraintCount, cache);
		}
		void solveExt1DBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(params);
			solveExt1DBlock(desc, constraintCount, cache);
		}
		void solveContact_BStaticBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(params);
			solveContact_BStaticBlock(desc, constraintCount, cache);
		}
		void solveContactPreBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(params);
			solveContactPreBlock(desc, constraintCount, cache);
		}
		void solveContactPreBlock_StaticLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(params);
			solveContactPreBlock_Static(desc, constraintCount, cache);
		}
		void solve1D4_BlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(params);
			solve1D4_Block(desc, constraintCount, cache);
		}

		static SolveBlockMethodLDL gVTableSolveBlockLDL[] = {
			0,
			solve1DBlockLDL,														// DY_SC_TYPE_RB_CONTACT
			solve1DBlockLDL,															// DY_SC_TYPE_RB_1D
			//DYNAMIC_ARTICULATION_REGISTRATIONLDL(solveExtContactBlock),				// DY_SC_TYPE_EXT_CONTACT
			//DYNAMIC_ARTICULATION_REGISTRATIONLDL(solveExt1DBlock),						// DY_SC_TYPE_EXT_1D
			solve1DBlockLDL,
			solve1DBlockLDL,
			solveContact_BStaticBlockLDL,												// DY_SC_TYPE_STATIC_CONTACT
			solveContactBlockLDL,														// DY_SC_TYPE_NOFRICTION_RB_CONTACT
			solveContactPreBlockLDL,													// DY_SC_TYPE_BLOCK_RB_CONTACT
			solveContactPreBlock_StaticLDL,											// DY_SC_TYPE_BLOCK_STATIC_RB_CONTACT
			solve1D4_BlockLDL,															// DY_SC_TYPE_BLOCK_1D,
		};

		static SolveBlockMethodLDL gVTableSolveWriteBackBlockLDL[] =
		{
			0,
			solveContactBlockLDL,														// DY_SC_TYPE_RB_CONTACT
			solve1DBlockLDL,															// DY_SC_TYPE_RB_1D
			solveExtContactBlockLDL,				// DY_SC_TYPE_EXT_CONTACT
			solveExt1DBlockLDL,						// DY_SC_TYPE_EXT_1D
			solveContact_BStaticBlockLDL,												// DY_SC_TYPE_STATIC_CONTACT
			solveContactBlockLDL,														// DY_SC_TYPE_NOFRICTION_RB_CONTACT
			solveContactPreBlockLDL,													// DY_SC_TYPE_BLOCK_RB_CONTACT
			solveContactPreBlock_StaticLDL,											// DY_SC_TYPE_BLOCK_STATIC_RB_CONTACT
			solve1D4_BlockLDL,															// DY_SC_TYPE_BLOCK_1D,
		};

		inline void SolveBlockParallelLDL(SolverIslandParams& params, PxSolverConstraintDesc* PX_RESTRICT constraintList, const PxI32 batchCount, const PxI32 index,
			const PxI32 headerCount, SolverContext& cache, BatchIterator& iterator,
			SolveBlockMethodLDL solveTable[],
			const PxI32 iteration
		)
		{
			const PxI32 indA = index - (iteration * headerCount);

			const PxConstraintBatchHeader* PX_RESTRICT headers = iterator.constraintBatchHeaders;

			const PxI32 endIndex = indA + batchCount;
			// 每个batch/header有不同的constraintType
			for (PxI32 i = indA; i < endIndex; ++i)
			{
				const PxConstraintBatchHeader& header = headers[i];

				const PxI32 numToGrab = header.stride;
				PxSolverConstraintDesc* PX_RESTRICT block = &constraintList[header.startIndex];

				Ps::prefetch(block[0].constraint, 384);

				for (PxI32 b = 0; b < numToGrab; ++b)
				{
					Ps::prefetchLine(block[b].bodyA);
					Ps::prefetchLine(block[b].bodyB);
				}

				//OK. We have a number of constraints to run...
				solveTable[header.constraintType](params, block, PxU32(numToGrab), cache);
			}
		}

		void solveContactLDL(const PxSolverConstraintDesc& desc, SolverContext& cache)
		{
			PxSolverBody& b0 = *desc.bodyA;
			PxSolverBody& b1 = *desc.bodyB;

			Vec3V linVel0 = V3LoadA(b0.linearVelocity);
			Vec3V linVel1 = V3LoadA(b1.linearVelocity);
			Vec3V angState0 = V3LoadA(b0.angularState);
			Vec3V angState1 = V3LoadA(b1.angularState);

			const PxU8* PX_RESTRICT last = desc.constraint + getConstraintLength(desc);

			//hopefully pointer aliasing doesn't bite.
			PxU8* PX_RESTRICT currPtr = desc.constraint;

			while (currPtr < last)
			{
				SolverContactHeader* PX_RESTRICT hdr = reinterpret_cast<SolverContactHeader*>(currPtr);
				currPtr += sizeof(SolverContactHeader);

				const PxU32 numNormalConstr = hdr->numNormalConstr;
				const PxU32	numFrictionConstr = hdr->numFrictionConstr;

				SolverContactPoint* PX_RESTRICT contacts = reinterpret_cast<SolverContactPoint*>(currPtr);
				Ps::prefetchLine(contacts);
				currPtr += numNormalConstr * sizeof(SolverContactPoint);

				PxF32* forceBuffer = reinterpret_cast<PxF32*>(currPtr);
				PX_UNUSED(forceBuffer);
				currPtr += sizeof(PxF32) * ((numNormalConstr + 3) & (~3));

				SolverContactFriction* PX_RESTRICT frictions = reinterpret_cast<SolverContactFriction*>(currPtr);
				currPtr += numFrictionConstr * sizeof(SolverContactFriction);

				const FloatV invMassA = FLoad(hdr->invMass0);
				const FloatV invMassB = FLoad(hdr->invMass1);

				const FloatV angDom0 = FLoad(hdr->angDom0);
				const FloatV angDom1 = FLoad(hdr->angDom1);

				const Vec3V contactNormal = Vec3V_From_Vec4V_WUndefined(hdr->normal_minAppliedImpulseForFrictionW);

				const FloatV accumulatedNormalImpulse = angDom0;// solveDynamicContacts(contacts, numNormalConstr, contactNormal, invMassA, invMassB,
					//angDom0, angDom1, linVel0, angState0, linVel1, angState1, forceBuffer);

				if (cache.doFriction && numFrictionConstr)
				{
					const FloatV staticFrictionCof = hdr->getStaticFriction();
					const FloatV dynamicFrictionCof = hdr->getDynamicFriction();
					const FloatV maxFrictionImpulse = FMul(staticFrictionCof, accumulatedNormalImpulse);
					const FloatV maxDynFrictionImpulse = FMul(dynamicFrictionCof, accumulatedNormalImpulse);
					const FloatV negMaxDynFrictionImpulse = FNeg(maxDynFrictionImpulse);

					BoolV broken = BFFFF();

					if (cache.writeBackIteration)
						Ps::prefetchLine(hdr->frictionBrokenWritebackByte);

					for (PxU32 i = 0; i < numFrictionConstr; i++)
					{
						SolverContactFriction& f = frictions[i];
						Ps::prefetchLine(&frictions[i], 128);


						const Vec4V normalXYZ_appliedForceW = f.normalXYZ_appliedForceW;
						const Vec4V raXnXYZ_velMultiplierW = f.raXnXYZ_velMultiplierW;
						const Vec4V rbXnXYZ_biasW = f.rbXnXYZ_biasW;

						const Vec3V normal = Vec3V_From_Vec4V(normalXYZ_appliedForceW);
						const Vec3V raXn = Vec3V_From_Vec4V(raXnXYZ_velMultiplierW);
						const Vec3V rbXn = Vec3V_From_Vec4V(rbXnXYZ_biasW);

						const FloatV appliedForce = V4GetW(normalXYZ_appliedForceW);
						const FloatV bias = V4GetW(rbXnXYZ_biasW);
						const FloatV velMultiplier = V4GetW(raXnXYZ_velMultiplierW);

						const FloatV targetVel = FLoad(f.targetVel);

						const Vec3V delLinVel0 = V3Scale(normal, invMassA);
						const Vec3V delLinVel1 = V3Scale(normal, invMassB);

						const Vec3V v0 = V3MulAdd(linVel0, normal, V3Mul(angState0, raXn));
						const Vec3V v1 = V3MulAdd(linVel1, normal, V3Mul(angState1, rbXn));
						const FloatV normalVel = V3SumElems(V3Sub(v0, v1));



						// appliedForce -bias * velMultiplier - a hoisted part of the total impulse computation
						const FloatV tmp1 = FNegScaleSub(FSub(bias, targetVel), velMultiplier, appliedForce);

						// Algorithm:
						// if abs(appliedForce + deltaF) > maxFrictionImpulse
						//    clamp newAppliedForce + deltaF to [-maxDynFrictionImpulse, maxDynFrictionImpulse]
						//      (i.e. clamp deltaF to [-maxDynFrictionImpulse-appliedForce, maxDynFrictionImpulse-appliedForce]
						//    set broken flag to true || broken flag

						// FloatV deltaF = FMul(FAdd(bias, normalVel), minusVelMultiplier);
						// FloatV potentialSumF = FAdd(appliedForce, deltaF);

						const FloatV totalImpulse = FNegScaleSub(normalVel, velMultiplier, tmp1);

						// On XBox this clamping code uses the vector simple pipe rather than vector float,
						// which eliminates a lot of stall cycles

						const BoolV clamp = FIsGrtr(FAbs(totalImpulse), maxFrictionImpulse);

						const FloatV totalClamped = FMin(maxDynFrictionImpulse, FMax(negMaxDynFrictionImpulse, totalImpulse));

						const FloatV newAppliedForce = FSel(clamp, totalClamped, totalImpulse);

						broken = BOr(broken, clamp);

						FloatV deltaF = FSub(newAppliedForce, appliedForce);

						// we could get rid of the stall here by calculating and clamping delta separately, but
						// the complexity isn't really worth it.

						linVel0 = V3ScaleAdd(delLinVel0, deltaF, linVel0);
						linVel1 = V3NegScaleSub(delLinVel1, deltaF, linVel1);
						angState0 = V3ScaleAdd(raXn, FMul(deltaF, angDom0), angState0);
						angState1 = V3NegScaleSub(rbXn, FMul(deltaF, angDom1), angState1);

						f.setAppliedForce(newAppliedForce);


					}
					Store_From_BoolV(broken, &hdr->broken);
				}

			}

			PX_ASSERT(b0.linearVelocity.isFinite());
			PX_ASSERT(b0.angularState.isFinite());
			PX_ASSERT(b1.linearVelocity.isFinite());
			PX_ASSERT(b1.angularState.isFinite());

			// Write back
			V3StoreU(linVel0, b0.linearVelocity);
			V3StoreU(linVel1, b1.linearVelocity);
			V3StoreU(angState0, b0.angularState);
			V3StoreU(angState1, b1.angularState);

			PX_ASSERT(b0.linearVelocity.isFinite());
			PX_ASSERT(b0.angularState.isFinite());
			PX_ASSERT(b1.linearVelocity.isFinite());
			PX_ASSERT(b1.angularState.isFinite());

			PX_ASSERT(currPtr == last);
		}

		void getAppliedImpulse(const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, VectorN& result, VectorN& resultSplit) {
			for (PxU32 a = 1; a < constraintCount; ++a)
			{
				Ps::prefetchLine(desc[a].constraint);
				Ps::prefetchLine(desc[a].constraint, 128);
				Ps::prefetchLine(desc[a].constraint, 256);

				// solve1D(desc[a - 1], cache);
				PxU8* PX_RESTRICT bPtr = desc[a - 1].constraint;
				if (bPtr == NULL)
					return;
				const SolverConstraint1DHeader* PX_RESTRICT  header = reinterpret_cast<const SolverConstraint1DHeader*>(bPtr);
				SolverConstraint1D* PX_RESTRICT base = reinterpret_cast<SolverConstraint1D*>(bPtr + sizeof(SolverConstraint1DHeader));

				for (PxU32 i = 0; i < header->count; ++i, base++)
				{
					Ps::prefetchLine(base + 1);
					SolverConstraint1D& c = *base;

					result[i] = c.impulseMultiplier;
					resultSplit[i] = c.impulseMultiplier;
				}
			}

			PxU8* PX_RESTRICT bPtr = desc[constraintCount - 1].constraint;
			if (bPtr == NULL)
				return;
			const SolverConstraint1DHeader* PX_RESTRICT  header = reinterpret_cast<const SolverConstraint1DHeader*>(bPtr);
			SolverConstraint1D* PX_RESTRICT base = reinterpret_cast<SolverConstraint1D*>(bPtr + sizeof(SolverConstraint1DHeader));

			for (PxU32 i = 0; i < header->count; ++i, base++)
			{
				Ps::prefetchLine(base + 1);
				SolverConstraint1D& c = *base;

				result[i] = c.impulseMultiplier;
				resultSplit[i] = c.impulseMultiplier;
			}
		}

		void solve1DBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PX_UNUSED(cache);
			//PxSolverBody* PX_RESTRICT bodyListStart = params.bodyListStart;
			const PxU32 bodyListSize = params.bodyListSize;

			PxU32 N = constraintCount;

			PxU8* PX_RESTRICT currPtr = desc->constraint;
			SolverContactHeader* PX_RESTRICT hdr = reinterpret_cast<SolverContactHeader*>(currPtr);

			MatrixNN A(N);
			VectorN b(N);
			VectorN bSplit(N);
			VectorN lo(N);
			VectorN hi(N);
			VectorN result(N);
			VectorN resultSplit(N);
			for (PxU32 i = 0; i < N; ++i) {
				lo[i] = 0x00;
				hi[i] = 0x7F800000;
			}

			int numContactRows = 1;

			PxU32 numConstraintRows = constraintCount;
			{
				for (PxU32 i = 0; i < numConstraintRows; i++)
				{
					PxU8* PX_RESTRICT bPtr = desc[i].constraint;
					if (bPtr == NULL)
						return;
					const SolverConstraint1DHeader* PX_RESTRICT  header = reinterpret_cast<const SolverConstraint1DHeader*>(bPtr + sizeof(SolverConstraint1DHeader));

					PxReal jacDiag = header->invMass0D0;
					if (jacDiag != 0) {
						PxReal rhs = header->invMass1D1;
						PxReal rhsPenetration = header->invMass1D1;
						b[i] = rhs / jacDiag;
						bSplit[i] = rhsPenetration / jacDiag;
					}
				}
			}

			//
			int m = constraintCount;

			int numBodies = bodyListSize;
			std::vector<int> bodyJointNodeArray(numBodies * 2, -1);
			std::vector<JointNode> jointNodeArray(2 * constraintCount);

			// MatrixNN& J3 = m_scratchJ3;
			MatrixMN J3(2 * m, 8);
			MatrixMN JinvM3(2 * m, 8);

			int cur = 0;
			int rowOffset = 0;
			std::vector<int> ofs(constraintCount);

			{
				// BT_PROFILE("Compute J and JinvM");
				int c = 0;

				int numRows = 0;

				for (PxU32 i = 0; i < constraintCount; i += numRows, c++)
				{
					ofs[c] = rowOffset;
					int sbA = desc->bodyADataIndex;
					int sbB = desc->bodyBDataIndex;

					numRows = (i < 0 ? numConstraintRows : numContactRows);
					if (1)
					{
						{
							int slotA = -1;
							//find free jointNode slot for sbA
							slotA = static_cast<int>(jointNodeArray.size());
							//jointNodeArray.expand();  //NonInitializing();
							//jointNodeArray.emplace_back(JointNode());
							int prevSlot = bodyJointNodeArray[sbA];
							bodyJointNodeArray[sbA] = slotA;
							jointNodeArray[slotA].nextJointNodeIndex = prevSlot;
							jointNodeArray[slotA].jointIndex = c;
							jointNodeArray[slotA].constraintRowIndex = i;
							jointNodeArray[slotA].otherBodyIndex = sbB;
						}
						for (int row = 0; row < numRows; row++, cur++)
						{
							PxVec3 normalInvMass;
							V3StoreU(Vec3V_From_Vec4V(hdr->normal_minAppliedImpulseForFrictionW), normalInvMass);
							normalInvMass *= hdr->invMass0;
							PxVec3 relPosCrossNormalInvInertia;
							V3StoreU(Vec3V_From_Vec4V(hdr->normal_minAppliedImpulseForFrictionW), relPosCrossNormalInvInertia);
							relPosCrossNormalInvInertia *= hdr->angDom0;

							for (int r = 0; r < 3; r++)
							{
								J3.set(cur, r, normalInvMass[r]);
								J3.set(cur, r + 4, normalInvMass[r]);
								JinvM3.set(cur, r, normalInvMass[r]);
								JinvM3.set(cur, r + 4, relPosCrossNormalInvInertia[r]);
							}
							J3.set(cur, 3, 0);
							JinvM3.set(cur, 3, 0);
							J3.set(cur, 7, 0);
							JinvM3.set(cur, 7, 0);
						}
					}
					else
					{
						cur += numRows;
					}
					if (1)
					{
						{
							int slotB = -1;
							//find free jointNode slot for sbA
							slotB = static_cast<int>(jointNodeArray.size());
							//jointNodeArray.expand();  //NonInitializing();
							//jointNodeArray.emplace_back(JointNode());
							int prevSlot = bodyJointNodeArray[sbB];
							bodyJointNodeArray[sbB] = slotB;
							jointNodeArray[slotB].nextJointNodeIndex = prevSlot;
							jointNodeArray[slotB].jointIndex = c;
							jointNodeArray[slotB].otherBodyIndex = sbA;
							jointNodeArray[slotB].constraintRowIndex = i;
						}

						for (int row = 0; row < numRows; row++, cur++)
						{
							PxVec3 normalInvMass;
							V3StoreU(Vec3V_From_Vec4V(hdr->normal_minAppliedImpulseForFrictionW), normalInvMass);
							normalInvMass *= hdr->invMass0;
							PxVec3 relPosCrossNormalInvInertia;
							V3StoreU(Vec3V_From_Vec4V(hdr->normal_minAppliedImpulseForFrictionW), relPosCrossNormalInvInertia);
							relPosCrossNormalInvInertia *= hdr->angDom0;

							for (int r = 0; r < 3; r++)
							{
								J3.set(cur, r, normalInvMass[r]);
								J3.set(cur, r + 4, normalInvMass[r]);
								JinvM3.set(cur, r, normalInvMass[r]);
								JinvM3.set(cur, r + 4, relPosCrossNormalInvInertia[r]);
							}
							J3.set(cur, 3, 0);
							JinvM3.set(cur, 3, 0);
							J3.set(cur, 7, 0);
							JinvM3.set(cur, 7, 0);
						}

						currPtr += sizeof(SolverContactHeader);
					}
					else
					{
						cur += numRows;
					}
					rowOffset += numRows;
				}
			}

			//compute JinvM = J*invM.
			const PxReal* JinvM = JinvM3.getBufferPointer();

			const PxReal* Jptr = J3.getBufferPointer();

			int c = 0;
			{
				int numRows = 0;
				//BT_PROFILE("Compute A");
				for (PxU32 i = 0; i < constraintCount; i += numRows, c++)
				{
					int row__ = ofs[c];
					int sbA = desc->bodyADataIndex;
					int sbB = desc->bodyBDataIndex;
					//	btRigidBody* orgBodyA = m_tmpSolverBodyPool[sbA].m_originalBody;
					//	btRigidBody* orgBodyB = m_tmpSolverBodyPool[sbB].m_originalBody;

					numRows = (i < 0 ? numConstraintRows : numContactRows);

					const PxReal* JinvMrow = JinvM + 2 * 8 * (size_t)row__;

					{
						int startJointNodeA = bodyJointNodeArray[sbA];
						while (startJointNodeA >= 0)
						{
							int j0 = jointNodeArray[startJointNodeA].jointIndex;
							int cr0 = jointNodeArray[startJointNodeA].constraintRowIndex;
							if (j0 < c)
							{
								int numRowsOther = (cr0 < 0 ? numConstraintRows : numContactRows);
								size_t ofsother = 8 * numRowsOther;
								//printf("%d joint i %d and j0: %d: ",count++,i,j0);
								A.multiplyAdd2_p8r(JinvMrow,
									Jptr + 2 * 8 * (size_t)ofs[j0] + ofsother, numRows, numRowsOther, row__, ofs[j0]);
							}
							startJointNodeA = jointNodeArray[startJointNodeA].nextJointNodeIndex;
						}
					}

					{
						int startJointNodeB = bodyJointNodeArray[sbB];
						while (startJointNodeB >= 0)
						{
							int j1 = jointNodeArray[startJointNodeB].jointIndex;
							int cj1 = jointNodeArray[startJointNodeB].constraintRowIndex;

							if (j1 < c)
							{
								int numRowsOther = (cj1 < 0 ? numConstraintRows : numContactRows);
								size_t ofsother =  8 * numRowsOther;
								A.multiplyAdd2_p8r(JinvMrow + 8 * (size_t)numRows,
									Jptr + 2 * 8 * (size_t)ofs[j1] + ofsother, numRows, numRowsOther, row__, ofs[j1]);
							}
							startJointNodeB = jointNodeArray[startJointNodeB].nextJointNodeIndex;
						}
					}
				}

				{
					// compute diagonal blocks of A

					int row__ = 0;
					int numJointRows = constraintCount;

					int jj = 0;
					for (; row__ < numJointRows;)
					{
						//PxSolverBody& b1 = *desc->bodyB;

						const unsigned int infom = (row__ < 0 ? numConstraintRows : numContactRows);

						const PxReal* JinvMrow = JinvM + 2 * 8 * (size_t)row__;
						const PxReal* Jrow = Jptr + 2 * 8 * (size_t)row__;
						A.multiply2_p8r(JinvMrow, Jrow, infom, infom, row__, row__);
						if (1)
						{
							A.multiply2_p8r(JinvMrow + 8 * (size_t)infom, Jrow + 8 * (size_t)infom, infom, infom, row__, row__);
						}
						row__ += infom;
						jj++;
					}
				}
			}

			if (1)
			{
				// add cfm to the diagonal of m_A
				const PxReal timestep = 1.0f / 60.0f;
				const PxReal cfm = 1e-4;
				for (PxU32 i = 0; i < A.getSize(); ++i) {
					A.set(i, i, A.get(i, i) + cfm / timestep);
				}
			}

			{
				// fill the upper triangle
				A.copyLowerToUpperTriangle();
			}

			getAppliedImpulse(desc, constraintCount, result, resultSplit);

			MatrixLDLGaussSeidelSolver::solve(1, A, b, lo, hi, result);

			for (PxU32 i = 0; i < numConstraintRows; ++i) {
				//PxSolverBody& b0 = *desc[i].bodyA;
				//PxSolverBody& b1 = *desc[i].bodyB;

				{
					PxReal deltaF = result[i];
					
					PxU8* PX_RESTRICT bPtr = desc[i].constraint;
					if (bPtr == NULL)
						return;
					//PxU32 length = desc.constraintLength;

					//const SolverConstraint1DHeader* PX_RESTRICT  header = reinterpret_cast<const SolverConstraint1DHeader*>(bPtr);
					SolverConstraint1D* PX_RESTRICT base = reinterpret_cast<SolverConstraint1D*>(bPtr + sizeof(SolverConstraint1DHeader));

					base->appliedForce = deltaF;

					solve1DBlock(desc, constraintCount, cache);
				}
			}
		}

		void solveContactBlockLDL(const SolverIslandParams& params, const PxSolverConstraintDesc* PX_RESTRICT desc, const PxU32 constraintCount, SolverContext& cache) {
			PxU32 N = params.bodyListSize;
			PX_UNUSED(N);
			for (PxU32 a = 1; a < constraintCount; ++a)
			{
				Ps::prefetchLine(desc[a].constraint);
				Ps::prefetchLine(desc[a].constraint, 128);
				Ps::prefetchLine(desc[a].constraint, 256);
				solveContactLDL(desc[a - 1], cache);
			}
			solveContactLDL(desc[constraintCount - 1], cache);
		}

		//--------------------------------------------------------

		SolverCoreLDL* SolverCoreLDL::create(bool fricEveryIteration)
		{
			SolverCoreLDL* scg = reinterpret_cast<SolverCoreLDL*>(
				PX_ALLOC(sizeof(SolverCoreLDL), "SolverCoreLDL"));

			if (scg)
			{
				new (scg) SolverCoreLDL;
				scg->frictionEveryIteration = fricEveryIteration;
			}

			return scg;
		}

		void SolverCoreLDL::destroyV()
		{
			this->~SolverCoreLDL();
			PX_FREE(this);
		}

		void SolverCoreLDL::solveV_Blocks(SolverIslandParams& params) const
		{
			const PxI32 TempThresholdStreamSize = 32;
			ThresholdStreamElement tempThresholdStream[TempThresholdStreamSize];

			SolverContext cache;
			cache.solverBodyArray = params.bodyDataList;
			cache.mThresholdStream = tempThresholdStream;
			cache.mThresholdStreamLength = TempThresholdStreamSize;
			cache.mThresholdStreamIndex = 0;
			cache.writeBackIteration = false;
			cache.Z = params.Z;
			cache.deltaV = params.deltaV;

			const PxI32 batchCount = PxI32(params.numConstraintHeaders);

			PxSolverBody* PX_RESTRICT bodyListStart = params.bodyListStart;
			const PxU32 bodyListSize = params.bodyListSize;

			Cm::SpatialVector* PX_RESTRICT motionVelocityArray = params.motionVelocityArray;

			const PxU32 velocityIterations = params.velocityIterations;
			const PxU32 positionIterations = params.positionIterations;

			const PxU32 numConstraintHeaders = params.numConstraintHeaders;
			const PxU32 articulationListSize = params.articulationListSize;

			ArticulationSolverDesc* PX_RESTRICT articulationListStart = params.articulationListStart;

			PX_ASSERT(velocityIterations >= 1);
			PX_ASSERT(positionIterations >= 1);

			if (numConstraintHeaders == 0)
			{
				for (PxU32 baIdx = 0; baIdx < bodyListSize; baIdx++)
				{
					Cm::SpatialVector& motionVel = motionVelocityArray[baIdx];
					const PxSolverBody& atom = bodyListStart[baIdx];

					motionVel.linear = atom.linearVelocity;
					motionVel.angular = atom.angularState;
				}

				//Even thought there are no external constraints, there may still be internal constraints in the articulations...
				for (PxU32 i = 0; i < positionIterations; ++i)
					for (PxU32 j = 0; j < articulationListSize; ++j)
						articulationListStart[j].articulation->solveInternalConstraints(params.dt, params.invDt, cache.Z, cache.deltaV, false, false, 0.f);

				for (PxU32 i = 0; i < articulationListSize; i++)
					ArticulationPImpl::saveVelocity(articulationListStart[i], cache.deltaV);

				for (PxU32 i = 0; i < velocityIterations; ++i)
					for (PxU32 j = 0; j < articulationListSize; ++j)
						articulationListStart[j].articulation->solveInternalConstraints(params.dt, params.invDt, cache.Z, cache.deltaV, true, false, 0.f);

				for (PxU32 j = 0; j < articulationListSize; ++j)
					articulationListStart[j].articulation->writebackInternalConstraints(false);

				return;
			}

			BatchIterator contactIterator(params.constraintBatchHeaders, params.numConstraintHeaders);

			PxSolverConstraintDesc* PX_RESTRICT constraintList = params.constraintList;

			//0-(n-1) iterations
			PxI32 normalIter = 0;

			for (PxU32 iteration = positionIterations; iteration > 0; iteration--)	//decreasing positive numbers == position iters
			{
				cache.doFriction = this->frictionEveryIteration ? true : iteration <= 3;

				SolveBlockParallelLDL(params, constraintList, batchCount, normalIter * batchCount, batchCount,
					cache, contactIterator, gVTableSolveBlockLDL, normalIter);

				for (PxU32 i = 0; i < articulationListSize; ++i)
					articulationListStart[i].articulation->solveInternalConstraints(params.dt, params.invDt, cache.Z, cache.deltaV, false, false, 0.f);

				++normalIter;
			}

			for (PxU32 baIdx = 0; baIdx < bodyListSize; baIdx++)
			{
				const PxSolverBody& atom = bodyListStart[baIdx];
				Cm::SpatialVector& motionVel = motionVelocityArray[baIdx];
				motionVel.linear = atom.linearVelocity;
				motionVel.angular = atom.angularState;
			}
			
			for (PxU32 i = 0; i < articulationListSize; i++)
				ArticulationPImpl::saveVelocity(articulationListStart[i], cache.deltaV);

			const PxI32 velItersMinOne = (PxI32(velocityIterations)) - 1;

			PxI32 iteration = 0;

			for (; iteration < velItersMinOne; ++iteration)
			{
				SolveBlockParallelLDL(params, constraintList, batchCount, normalIter * batchCount, batchCount,
					cache, contactIterator, gVTableSolveBlockLDL, normalIter);

				for (PxU32 i = 0; i < articulationListSize; ++i)
					articulationListStart[i].articulation->solveInternalConstraints(params.dt, params.invDt, cache.Z, cache.deltaV, true, false, 0.f);
				++normalIter;
			}

			PxI32* outThresholdPairs = params.outThresholdPairs;
			ThresholdStreamElement* PX_RESTRICT thresholdStream = params.thresholdStream;
			PxU32 thresholdStreamLength = params.thresholdStreamLength;

			cache.writeBackIteration = true;
			cache.mSharedThresholdStream = thresholdStream;
			cache.mSharedThresholdStreamLength = thresholdStreamLength;
			cache.mSharedOutThresholdPairs = outThresholdPairs;
			//PGS solver always runs at least one velocity iteration (otherwise writeback won't happen)
			{
				SolveBlockParallelLDL(params, constraintList, batchCount, normalIter * batchCount, batchCount,
					cache, contactIterator, gVTableSolveWriteBackBlockLDL, normalIter);

				for (PxU32 i = 0; i < articulationListSize; ++i)
				{
					articulationListStart[i].articulation->solveInternalConstraints(params.dt, params.invDt, cache.Z, cache.deltaV, true, false, 0.f);
					articulationListStart[i].articulation->writebackInternalConstraints(false);
				}

				++normalIter;
			}

			//Write back remaining threshold streams
			if (cache.mThresholdStreamIndex > 0)
			{
				//Write back to global buffer
				PxI32 threshIndex = physx::shdfnd::atomicAdd(outThresholdPairs, PxI32(cache.mThresholdStreamIndex)) - PxI32(cache.mThresholdStreamIndex);
				for (PxU32 b = 0; b < cache.mThresholdStreamIndex; ++b)
				{
					thresholdStream[b + threshIndex] = cache.mThresholdStream[b];
				}
				cache.mThresholdStreamIndex = 0;
			}
		}

		PxI32 SolverCoreLDL::solveVParallelAndWriteBack
		(SolverIslandParams& params, Cm::SpatialVectorF* Z, Cm::SpatialVectorF* deltaV) const
		{
#if PX_PROFILE_SOLVE_STALLS
			PxU64 startTime = readTimer();

			PxU64 stallCount = 0;
#endif

			SolverContext cache;
			cache.solverBodyArray = params.bodyDataList;
			const PxU32 batchSize = params.batchSize;

			const PxI32 UnrollCount = PxI32(batchSize);
			const PxI32 ArticCount = 2;
			const PxI32 SaveUnrollCount = 32;

			const PxI32 TempThresholdStreamSize = 32;
			ThresholdStreamElement tempThresholdStream[TempThresholdStreamSize];

			const PxI32 bodyListSize = PxI32(params.bodyListSize);
			const PxI32 articulationListSize = PxI32(params.articulationListSize);

			const PxI32 batchCount = PxI32(params.numConstraintHeaders);
			cache.mThresholdStream = tempThresholdStream;
			cache.mThresholdStreamLength = TempThresholdStreamSize;
			cache.mThresholdStreamIndex = 0;
			cache.writeBackIteration = false;
			cache.Z = Z;
			cache.deltaV = deltaV;

			const PxReal dt = params.dt;
			const PxReal invDt = params.invDt;

			const PxI32 positionIterations = PxI32(params.positionIterations);
			const PxI32 velocityIterations = PxI32(params.velocityIterations);

			PxI32* constraintIndex = &params.constraintIndex;
			PxI32* constraintIndex2 = &params.constraintIndex2;

			PxI32* articIndex = &params.articSolveIndex;
			PxI32* articIndex2 = &params.articSolveIndex2;

			//PxSolverConstraintDesc* PX_RESTRICT constraintList = params.constraintList;

			ArticulationSolverDesc* PX_RESTRICT articulationListStart = params.articulationListStart;

			const PxU32 nbPartitions = params.nbPartitions;

			PxU32* headersPerPartition = params.headersPerPartition;

			PX_UNUSED(velocityIterations);

			PX_ASSERT(velocityIterations >= 1);
			PX_ASSERT(positionIterations >= 1);

			PxI32 endIndexCount = UnrollCount;
			PxI32 index = physx::shdfnd::atomicAdd(constraintIndex, UnrollCount) - UnrollCount;

			PxI32 articSolveStart = 0;
			PxI32 articSolveEnd = 0;
			PxI32 maxArticIndex = 0;
			PxI32 articIndexCounter = 0;

			BatchIterator contactIter(params.constraintBatchHeaders, params.numConstraintHeaders);

			PxI32 maxNormalIndex = 0;
			PxI32 normalIteration = 0;
			PxU32 a = 0;
			PxI32 targetConstraintIndex = 0;
			PxI32 targetArticIndex = 0;

			for (PxU32 i = 0; i < 2; ++i)
			{
				//SolveBlockMethod* solveTable = i == 0 ? gVTableSolveBlock : gVTableSolveConcludeBlock;
				for (; a < positionIterations - 1 + i; ++a)
				{
					WAIT_FOR_PROGRESS(articIndex2, targetArticIndex);

					cache.doFriction = this->frictionEveryIteration ? true : (positionIterations - a) <= 3;
					for (PxU32 b = 0; b < nbPartitions; ++b)
					{
						WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);

						maxNormalIndex += headersPerPartition[b];

						PxI32 nbSolved = 0;
						while (index < maxNormalIndex)
						{
							const PxI32 remainder = PxMin(maxNormalIndex - index, endIndexCount);
							//SolveBlockParallelLDL(params, constraintList, remainder, index, batchCount, cache, contactIter, gVTableSolveBlockLDL,
							//	normalIteration);
							index += remainder;
							endIndexCount -= remainder;
							nbSolved += remainder;
							if (endIndexCount == 0)
							{
								endIndexCount = UnrollCount;
								index = physx::shdfnd::atomicAdd(constraintIndex, UnrollCount) - UnrollCount;
							}
						}
						if (nbSolved)
						{
							Ps::memoryBarrier();
							physx::shdfnd::atomicAdd(constraintIndex2, nbSolved);
						}
						targetConstraintIndex += headersPerPartition[b]; //Increment target constraint index by batch count
					}

					WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);

					maxArticIndex += articulationListSize;
					targetArticIndex += articulationListSize;

					while (articSolveStart < maxArticIndex)
					{
						const PxI32 endIdx = PxMin(articSolveEnd, maxArticIndex);

						PxI32 nbSolved = 0;
						while (articSolveStart < endIdx)
						{
							articulationListStart[articSolveStart - articIndexCounter].articulation->solveInternalConstraints(dt, invDt, cache.Z, cache.deltaV, false, false, 0.f);
							articSolveStart++;
							nbSolved++;
						}

						if (nbSolved)
						{
							physx::shdfnd::atomicAdd(articIndex2, nbSolved);
						}

						const PxI32 remaining = articSolveEnd - articSolveStart;

						if (remaining == 0)
						{
							articSolveStart = physx::shdfnd::atomicAdd(articIndex, ArticCount) - ArticCount;
							articSolveEnd = articSolveStart + ArticCount;
						}
					}

					articIndexCounter += articulationListSize;

					++normalIteration;
				}
			}

			PxI32* bodyListIndex = &params.bodyListIndex;
			PxI32* bodyListIndex2 = &params.bodyListIndex2;

			PxSolverBody* PX_RESTRICT bodyListStart = params.bodyListStart;
			Cm::SpatialVector* PX_RESTRICT motionVelocityArray = params.motionVelocityArray;

			//Save velocity - articulated
			PxI32 endIndexCount2 = SaveUnrollCount;
			PxI32 index2 = physx::shdfnd::atomicAdd(bodyListIndex, SaveUnrollCount) - SaveUnrollCount;
			{
				WAIT_FOR_PROGRESS(articIndex2, targetArticIndex);
				WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);
				PxI32 nbConcluded = 0;
				while (index2 < articulationListSize)
				{
					const PxI32 remainder = PxMin(SaveUnrollCount, (articulationListSize - index2));
					endIndexCount2 -= remainder;
					for (PxI32 b = 0; b < remainder; ++b, ++index2)
					{
						ArticulationPImpl::saveVelocity(articulationListStart[index2], cache.deltaV);
					}
					if (endIndexCount2 == 0)
					{
						index2 = physx::shdfnd::atomicAdd(bodyListIndex, SaveUnrollCount) - SaveUnrollCount;
						endIndexCount2 = SaveUnrollCount;
					}
					nbConcluded += remainder;
				}

				index2 -= articulationListSize;

				//save velocity

				while (index2 < bodyListSize)
				{
					const PxI32 remainder = PxMin(endIndexCount2, (bodyListSize - index2));
					endIndexCount2 -= remainder;
					for (PxI32 b = 0; b < remainder; ++b, ++index2)
					{
						Ps::prefetchLine(&bodyListStart[index2 + 8]);
						Ps::prefetchLine(&motionVelocityArray[index2 + 8]);
						PxSolverBody& body = bodyListStart[index2];
						Cm::SpatialVector& motionVel = motionVelocityArray[index2];
						motionVel.linear = body.linearVelocity;
						motionVel.angular = body.angularState;
						PX_ASSERT(motionVel.linear.isFinite());
						PX_ASSERT(motionVel.angular.isFinite());
					}

					nbConcluded += remainder;

					//Branch not required because this is the last time we use this atomic variable
					//if(index2 < articulationListSizePlusbodyListSize)
					{
						index2 = physx::shdfnd::atomicAdd(bodyListIndex, SaveUnrollCount) - SaveUnrollCount - articulationListSize;
						endIndexCount2 = SaveUnrollCount;
					}
				}

				if (nbConcluded)
				{
					Ps::memoryBarrier();
					physx::shdfnd::atomicAdd(bodyListIndex2, nbConcluded);
				}
			}

			WAIT_FOR_PROGRESS(bodyListIndex2, (bodyListSize + articulationListSize));

			a = 1;
			for (; a < params.velocityIterations; ++a)
			{
				WAIT_FOR_PROGRESS(articIndex2, targetArticIndex);
				for (PxU32 b = 0; b < nbPartitions; ++b)
				{
					WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);

					maxNormalIndex += headersPerPartition[b];

					PxI32 nbSolved = 0;
					while (index < maxNormalIndex)
					{
						const PxI32 remainder = PxMin(maxNormalIndex - index, endIndexCount);
						//SolveBlockParallel(constraintList, remainder, index, batchCount, cache, contactIter, gVTableSolveBlock,
						//	normalIteration);

						index += remainder;
						endIndexCount -= remainder;
						nbSolved += remainder;
						if (endIndexCount == 0)
						{
							endIndexCount = UnrollCount;
							index = physx::shdfnd::atomicAdd(constraintIndex, UnrollCount) - UnrollCount;
						}
					}
					if (nbSolved)
					{
						Ps::memoryBarrier();
						physx::shdfnd::atomicAdd(constraintIndex2, nbSolved);
					}
					targetConstraintIndex += headersPerPartition[b]; //Increment target constraint index by batch count
				}

				WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);

				maxArticIndex += articulationListSize;
				targetArticIndex += articulationListSize;

				while (articSolveStart < maxArticIndex)
				{
					const PxI32 endIdx = PxMin(articSolveEnd, maxArticIndex);

					PxI32 nbSolved = 0;
					while (articSolveStart < endIdx)
					{
						articulationListStart[articSolveStart - articIndexCounter].articulation->solveInternalConstraints(dt, invDt, cache.Z, cache.deltaV, true, false, 0.f);
						articSolveStart++;
						nbSolved++;
					}

					if (nbSolved)
					{
						physx::shdfnd::atomicAdd(articIndex2, nbSolved);
					}

					const PxI32 remaining = articSolveEnd - articSolveStart;

					if (remaining == 0)
					{
						articSolveStart = physx::shdfnd::atomicAdd(articIndex, ArticCount) - ArticCount;
						articSolveEnd = articSolveStart + ArticCount;
					}

				}
				++normalIteration;
				articIndexCounter += articulationListSize;
			}

			ThresholdStreamElement* PX_RESTRICT thresholdStream = params.thresholdStream;
			PxU32 thresholdStreamLength = params.thresholdStreamLength;
			PxI32* outThresholdPairs = params.outThresholdPairs;

			cache.mSharedOutThresholdPairs = outThresholdPairs;
			cache.mSharedThresholdStream = thresholdStream;
			cache.mSharedThresholdStreamLength = thresholdStreamLength;

			//Last iteration - do writeback as well!
			cache.writeBackIteration = true;
			{
				WAIT_FOR_PROGRESS(articIndex2, targetArticIndex);
				for (PxU32 b = 0; b < nbPartitions; ++b)
				{
					WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);

					maxNormalIndex += headersPerPartition[b];

					PxI32 nbSolved = 0;
					while (index < maxNormalIndex)
					{
						const PxI32 remainder = PxMin(maxNormalIndex - index, endIndexCount);

						//SolveBlockParallel(constraintList, remainder, index, batchCount, cache, contactIter, gVTableSolveWriteBackBlock,
						//	normalIteration);

						index += remainder;
						endIndexCount -= remainder;
						nbSolved += remainder;
						if (endIndexCount == 0)
						{
							endIndexCount = UnrollCount;
							index = physx::shdfnd::atomicAdd(constraintIndex, UnrollCount) - UnrollCount;
						}
					}
					if (nbSolved)
					{
						Ps::memoryBarrier();
						physx::shdfnd::atomicAdd(constraintIndex2, nbSolved);
					}
					targetConstraintIndex += headersPerPartition[b]; //Increment target constraint index by batch count
				}
				{
					WAIT_FOR_PROGRESS(constraintIndex2, targetConstraintIndex);

					maxArticIndex += articulationListSize;
					targetArticIndex += articulationListSize;

					while (articSolveStart < maxArticIndex)
					{
						const PxI32 endIdx = PxMin(articSolveEnd, maxArticIndex);

						PxI32 nbSolved = 0;
						while (articSolveStart < endIdx)
						{
							articulationListStart[articSolveStart - articIndexCounter].articulation->solveInternalConstraints(dt, invDt, cache.Z, cache.deltaV, false, false, 0.f);
							articulationListStart[articSolveStart - articIndexCounter].articulation->writebackInternalConstraints(false);
							articSolveStart++;
							nbSolved++;
						}

						if (nbSolved)
						{
							physx::shdfnd::atomicAdd(articIndex2, nbSolved);
						}

						PxI32 remaining = articSolveEnd - articSolveStart;

						if (remaining == 0)
						{
							articSolveStart = physx::shdfnd::atomicAdd(articIndex, ArticCount) - ArticCount;
							articSolveEnd = articSolveStart + ArticCount;
						}
					}
				}

				if (cache.mThresholdStreamIndex > 0)
				{
					//Write back to global buffer
					PxI32 threshIndex = physx::shdfnd::atomicAdd(outThresholdPairs, PxI32(cache.mThresholdStreamIndex)) - PxI32(cache.mThresholdStreamIndex);
					for (PxU32 b = 0; b < cache.mThresholdStreamIndex; ++b)
					{
						thresholdStream[b + threshIndex] = cache.mThresholdStream[b];
					}
					cache.mThresholdStreamIndex = 0;
				}

				++normalIteration;
			}

#if PX_PROFILE_SOLVE_STALLS

			PxU64 endTime = readTimer();
			PxReal totalTime = (PxReal)(endTime - startTime);
			PxReal stallTime = (PxReal)stallCount;
			PxReal stallRatio = stallTime / totalTime;
			if (0)//stallRatio > 0.2f)
			{
				LARGE_INTEGER frequency;
				QueryPerformanceFrequency(&frequency);
				printf("Warning -- percentage time stalled = %f; stalled for %f seconds; total Time took %f seconds\n",
					stallRatio * 100.f, stallTime / (PxReal)frequency.QuadPart, totalTime / (PxReal)frequency.QuadPart);
			}
#endif

			return normalIteration * batchCount;
		}

		void SolverCoreLDL::writeBackV
		(const PxSolverConstraintDesc* PX_RESTRICT constraintList, const PxU32 /*constraintListSize*/, PxConstraintBatchHeader* batchHeaders, const PxU32 numBatches,
			ThresholdStreamElement* PX_RESTRICT thresholdStream, const PxU32 thresholdStreamLength, PxU32& outThresholdPairs,
			PxSolverBodyData* atomListData, WriteBackBlockMethod writeBackTable[]) const
		{
			SolverContext cache;
			cache.solverBodyArray = atomListData;
			cache.mThresholdStream = thresholdStream;
			cache.mThresholdStreamLength = thresholdStreamLength;
			cache.mThresholdStreamIndex = 0;

			PxI32 outThreshIndex = 0;
			for (PxU32 j = 0; j < numBatches; ++j)
			{
				PxU8 type = *constraintList[batchHeaders[j].startIndex].constraint;
				writeBackTable[type](constraintList + batchHeaders[j].startIndex,
					batchHeaders[j].stride, cache);
			}

			outThresholdPairs = PxU32(outThreshIndex);
		}

	}
}