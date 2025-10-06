/*
 * XOR MLP
 *
 * o = xor(a, b)
 *
 * +---+---++---+
 * | a | b || o |
 * +---+---++---+
 * | 0 | 0 || 0 |
 * | 0 | 1 || 1 |
 * | 1 | 0 || 1 |
 * | 1 | 1 || 0 |
 * +---+---++---+
 *
 * Inputs:
 *   a and b flattened (2 parameters)
 * Output:
 *   o = xor(a, b)      (1 parameter)
 *
             inputs     HL    output
                  o.....o.
                    ....o....
                   .....o.....o
                  o.....o..
               [layers: 2 × 4 × 1]
 */

#include <stdio.h>
#include <array>

/* Macro Jail */
typedef float f32;
/* --- */

/* Const Jail */
constexpr size_t INP_LAYERS = 2;
constexpr size_t OUT_LAYERS = 1;
constexpr size_t          N = 4;  // number of training examples
/* --- */

/* Prototype Jail */
/* --- */

inline constexpr std::array<std::array<f32, INP_LAYERS>, N> inp_data = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};
inline constexpr std::array<std::array<f32, OUT_LAYERS>, N> out_data = {
    {0},
    {1},
    {1},
    {0}
};

int main()
{

    for (size_t i=0; i < N; i++)
    {
        printf("(%.2f, %.2f) = %.2f\n", inp_data[i][0], inp_data[i][1], out_data[i][0]);
    }
    return 0;
}
