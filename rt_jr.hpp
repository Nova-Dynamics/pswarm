#ifndef INCLUDE_PSWARM_RT_JR_HPP
#define INCLUDE_PSWARM_RT_JR_HPP

#include <cmath>

namespace rt_jr {

    template<typename T1, typename T2>
    struct Pair {
        T1 first;
        T2 second;
    };

    // functions and ops defined in structs are implicitly inlined
    struct Vec2 {
        float x, y;

        // const here just means that this method doesn't modify the object
        // that it's called on.
        __host__ __device__ Vec2 operator+(Vec2 v) const { return { x + v.x, y + v.y }; }
        __host__ __device__ Vec2 operator-(Vec2 v) const { return { x - v.x, y - v.y }; }
        __host__ __device__ Vec2 operator*(float s) const { return { x * s, y * s }; }
        __host__ __device__ float dot(Vec2 v) const { return x * v.x + y * v.y; }
        __host__ __device__ float cross(Vec2 v) const {
            return x * v.y - y * v.x;
        }
        __host__ __device__ float length() const { return sqrtf(x * x + y * y); }
    };

    struct Mat2 {
        float m[4]; // Let's choose row-major

        __host__ __device__ float& operator()(int r, int c) { return m[r * 2 + c]; } // Used for modification/assignment (row-major)
        __host__ __device__ const float& operator()(int r, int c) const { return m[r * 2 + c]; } // Used for read-only reference

        __host__ __device__ Mat2 transpose() const {
            Mat2 result;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    result(i, j) = (*this)(j, i);
                }
            }

            return result;
        }
    };

    // row-major matrix multiplication
    __host__ __device__ inline Mat2 operator*(const Mat2& a, const Mat2& b) {
        Mat2 result;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                result(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j);
            }
        }

        return result;
    }

    // row-major matrix-vector multiplication
    __host__ __device__ inline Vec2 operator*(const Mat2& a, const Vec2& v) {
        Vec2 result{};
        result.x = a(0, 0) * v.x + a(0, 1) * v.y;
        result.y = a(1, 0) * v.x + a(1, 1) * v.y;

        return result;
    }

    struct Vec3 {
        float x, y, z;

        __host__ __device__ Vec3 operator+(Vec3 v) const { return { x + v.x, y + v.y, z + v.z }; }
        __host__ __device__ Vec3 operator-(Vec3 v) const { return { x - v.x, y - v.y, z - v.z }; }
        __host__ __device__ Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
        __host__ __device__ float dot(Vec3 v) const { return x * v.x + y * v.y + z * v.z; }
        __host__ __device__ Vec3 cross(Vec3 v) const {
            return { y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x };
        }
        __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    };

    struct Mat3 {
        float m[9];

        __host__ __device__ float& operator()(int r, int c) { return m[r * 3 + c]; }
        __host__ __device__ const float& operator()(int r, int c) const { return m[r * 3 + c]; }

        __host__ __device__ Mat3 transpose() const {
            Mat3 result;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result(i, j) = (*this)(j, i);
                }
            }

            return result;
        }
    };

    __host__ __device__ inline Mat3 operator*(const Mat3& a, const Mat3& b) {
        Mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
            }
        }

        return result;
    }

    __host__ __device__ inline Vec3 operator*(const Mat3& a, const Vec3& v) {
        Vec3 result{};
        result.x = a(0, 0) * v.x + a(0, 1) * v.y + a(0, 2) * v.z;
        result.y = a(1, 0) * v.x + a(1, 1) * v.y + a(1, 2) * v.z;
        result.z = a(2, 0) * v.x + a(2, 1) * v.y + a(2, 2) * v.z;

        return result;
    }

    __host__ __device__ inline Mat2 get_R_from_theta(float theta) {
        Mat2 R{};

        float c = cosf(theta);
        float s = sinf(theta);
        R(0, 0) = c;
        R(1, 0) = s;
        R(0, 1) = -s;
        R(1, 1) = c;

        return R;
    }

    __host__ __device__ inline Pair<Mat2, Vec2> get_affine_tx_from_state(Vec3 state) {
        Mat2 R{};
        Vec2 t{};

        // Populate translation vector
        t.x = state.x;
        t.y = state.y;

        // NOTE: because state vector theta represents
        // clockwise rotations, need to keep this in mind
        // when using R downstream. This usually just looks
        // like transposing R before using in practice.
        R = get_R_from_theta(state.z);

        return { R, t };
    }

    __host__ __device__ inline Vec2 quantize_pt(Vec2 pt, float cell_width) {
        Vec2 qpt{};

        qpt.x = roundf(pt.x / cell_width);
        qpt.y = roundf(pt.y / cell_width);

        return qpt;
    }

    __host__ __device__ inline Vec2 body2map(Vec3 body_state_map, Vec2 body_pt) {
        Mat2 R{};
        Vec2 body_state_pos{ body_state_map.x, body_state_map.y };
        Vec2 result{};

        R = get_R_from_theta(body_state_map.z);
        result = body_state_pos + R.transpose() * body_pt;

        return result;
    }

}

#endif // INCLUDE_PSWARM_RT_JR_HPP
