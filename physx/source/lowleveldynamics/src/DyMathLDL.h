#ifndef DY_MATH_LDL_H 
#define DY_MATH_LDL_H

#include "PxPhysicsAPI.h"

namespace physx
{

#define MAX_VECTORN_SIZE 30 

	class VectorN
	{
	public:

		VectorN(const PxU32 size)
			: mSize(size)
		{
			PX_ASSERT(mSize <= MAX_VECTORN_SIZE);
		}
		~VectorN()
		{
		}

		VectorN(const VectorN& src)
		{
			for (PxU32 i = 0; i < src.mSize; i++)
			{
				mValues[i] = src.mValues[i];
			}
			mSize = src.mSize;
		}

		PX_FORCE_INLINE VectorN& operator=(const VectorN& src)
		{
			for (PxU32 i = 0; i < src.mSize; i++)
			{
				mValues[i] = src.mValues[i];
			}
			mSize = src.mSize;
			return *this;
		}

		PX_FORCE_INLINE PxF32& operator[] (const PxU32 i)
		{
			PX_ASSERT(i < mSize);
			return (mValues[i]);
		}

		PX_FORCE_INLINE const PxF32& operator[] (const PxU32 i) const
		{
			PX_ASSERT(i < mSize);
			return (mValues[i]);
		}

		PX_FORCE_INLINE PxU32 getSize() const { return mSize; }

	private:

		PxF32 mValues[MAX_VECTORN_SIZE];
		PxU32 mSize;
	};

	class MatrixNN
	{
	public:

		MatrixNN()
			: mSize(0)
		{
		}
		MatrixNN(const PxU32 size)
			: mSize(size)
		{
			PX_ASSERT(mSize <= MAX_VECTORN_SIZE);
		}
		MatrixNN(const MatrixNN& src)
		{
			for (PxU32 i = 0; i < src.mSize; i++)
			{
				for (PxU32 j = 0; j < src.mSize; j++)
				{
					mValues[i][j] = src.mValues[i][j];
				}
			}
			mSize = src.mSize;
		}
		~MatrixNN()
		{
		}

		PX_FORCE_INLINE MatrixNN& operator=(const MatrixNN& src)
		{
			for (PxU32 i = 0; i < src.mSize; i++)
			{
				for (PxU32 j = 0; j < src.mSize; j++)
				{
					mValues[i][j] = src.mValues[i][j];
				}
			}
			mSize = src.mSize;
			return *this;
		}

		PX_FORCE_INLINE PxF32 get(const PxU32 i, const PxU32 j) const
		{
			PX_ASSERT(i < mSize);
			PX_ASSERT(j < mSize);
			return mValues[i][j];
		}
		PX_FORCE_INLINE void set(const PxU32 i, const PxU32 j, const PxF32 val)
		{
			if (i >= mSize || j >= mSize) return;
			//PX_ASSERT(i < mSize);
			//PX_ASSERT(j < mSize);
			mValues[i][j] = val;
		}

		PX_FORCE_INLINE PxU32 getSize() const { return mSize; }

		PX_FORCE_INLINE void setSize(const PxU32 size)
		{
			PX_ASSERT(size <= MAX_VECTORN_SIZE);
			mSize = size;
		}

		PX_FORCE_INLINE void copyLowerToUpperTriangle() {
			for (PxU32 i = 0; i < mSize; i++)
			{
				for (PxU32 j = i + 1; j < mSize; j++)
				{
					mValues[i][j] = mValues[j][i];
				}
			}
		}

		PX_FORCE_INLINE void multiplyAdd2_p8r(const PxReal* B, const PxReal* C, int numRows, int numRowsOther, int row, int col)
		{
			const PxReal* bb = B;
			for (int i = 0; i < numRows; i++)
			{
				const PxReal* cc = C;
				for (int j = 0; j < numRowsOther; j++)
				{
					PxReal sum;
					sum = bb[0] * cc[0];
					sum += bb[1] * cc[1];
					sum += bb[2] * cc[2];
					sum += bb[4] * cc[4];
					sum += bb[5] * cc[5];
					sum += bb[6] * cc[6];
					set(row + i, col + j, sum);
					cc += 8;
				}
				bb += 8;
			}
		}

		PX_FORCE_INLINE void multiply2_p8r(const PxReal* B, const PxReal* C, int numRows, int numRowsOther, int row, int col)
		{
			//PX_ASSERT(numRows > 0 && numRowsOther > 0 && B && C);
			const PxReal* bb = B;
			for (int i = 0; i < numRows; i++)
			{
				const PxReal* cc = C;
				for (int j = 0; j < numRowsOther; j++)
				{
					PxReal sum;
					sum = bb[0] * cc[0];
					sum += bb[1] * cc[1];
					sum += bb[2] * cc[2];
					sum += bb[4] * cc[4];
					sum += bb[5] * cc[5];
					sum += bb[6] * cc[6];
					set(row + i, col + j, sum);
					cc += 8;
				}
				bb += 8;
			}
		}

		const PxReal* getBufferPointer() const
		{
			return mValues[0];
		}

	public:

		PxF32 mValues[MAX_VECTORN_SIZE][MAX_VECTORN_SIZE];
		PxU32 mSize;
	};

	class MatrixMN
	{
	public:

		MatrixMN()
			: mRow(0), mCol(0)
		{
		}
		MatrixMN(const PxU32 row, const PxU32 col)
			: mRow(row), mCol(col)
		{
			PX_ASSERT(mRow <= MAX_VECTORN_SIZE);
			PX_ASSERT(mCol <= MAX_VECTORN_SIZE);
		}
		MatrixMN(const MatrixMN& src)
		{
			for (PxU32 i = 0; i < src.mRow; i++)
			{
				for (PxU32 j = 0; j < src.mCol; j++)
				{
					mValues[i][j] = src.mValues[i][j];
				}
			}
			mRow = src.mRow;
			mCol = src.mCol;
		}
		~MatrixMN()
		{
		}

		PX_FORCE_INLINE MatrixMN& operator=(const MatrixMN& src)
		{
			for (PxU32 i = 0; i < src.mRow; i++)
			{
				for (PxU32 j = 0; j < src.mCol; j++)
				{
					mValues[i][j] = src.mValues[i][j];
				}
			}
			mRow = src.mRow;
			mCol = src.mCol;
			return *this;
		}

		PX_FORCE_INLINE PxF32 get(const PxU32 i, const PxU32 j) const
		{
			PX_ASSERT(i < mRow);
			PX_ASSERT(j < mCol);
			return mValues[i][j];
		}
		PX_FORCE_INLINE void set(const PxU32 i, const PxU32 j, const PxF32 val)
		{
			PX_ASSERT(i < mRow);
			PX_ASSERT(j < mCol);
			mValues[i][j] = val;
		}

		PX_FORCE_INLINE PxU32 getRow() const { return mRow; }
		PX_FORCE_INLINE PxU32 getCol() const { return mCol; }

		PX_FORCE_INLINE void setSize(const PxU32 row, const PxU32 col)
		{
			PX_ASSERT(mRow <= MAX_VECTORN_SIZE);
			mRow = row;
			mCol = col;
		}

		const PxReal* getBufferPointer() const
		{
			return mValues[0];
		}
	public:

		PxF32 mValues[MAX_VECTORN_SIZE][MAX_VECTORN_SIZE];
		PxU32 mRow;
		PxU32 mCol;
	};

	class MatrixLDLGaussSeidelSolver {
	public:
		static bool solve(const PxU32 maxIterations, const MatrixNN &A, const VectorN &b, const VectorN& lo, const VectorN& hi, VectorN &result) {
			const PxU32 N = A.getSize();
			if (N != b.getSize() || N != result.getSize()) {
				// output log
				return false;
			}
			const PxF32 eps = 1e-6f;
			VectorN invDial(N);
			for (PxU32 i = 0; i < N; ++i) {
				invDial[i] = 1.f / A.get(i, i);
			}

			for (PxU32 k = 0; k < maxIterations; k++) { // for each iteration
				PxReal squareResidual = 0.f; // error
				for (PxU32 i = 0; i < N; i++) { // for each row, calc A(i, j) * b[j]
					PxReal oldresult = result[i];

					PxF32 sum = 0.0f;
					for (PxU32 j = 0; j < i; j++) { 
						sum += A.get(i, j) * result[j];
					}

					for (PxU32 j = i + 1; j < N; j++) {
						sum += A.get(i, j) * result[j];
					}

					result[i] = (b[i] - sum) * invDial[i];

					if (result[i] < lo[i]) {
						result[i] = lo[i];
					}
					if (result[i] > hi[i]) {
						result[i] = hi[i];
					}

					PxReal diff = (result[i] - oldresult);
					squareResidual += diff * diff;
				}
				
				if (squareResidual < eps) {
					break;
				}
			}

			return true;
		}
	};
}

#endif // 
