.text

.globl bit_shift
bit_shift: // uint64_t bit_shift(uint64_t value, int exponent);
// Value is in d0, exponent in r2
    // Shift LSB by (32  - exp)
    lslx r3, r1, r2, sh32, .non_zero

    // If no jump, perform additional shift
    lsl r0, r0, r2 // r0 = value_h << exp
    lsl r1, r1, r2 // r1 = value_l << exp
    or r0, r0, r3 // r0 = (value_h << exp) | (value_l >> (32-exp))
    jump r23

    .non_zero:
    lsl r0, r1, r2 // r0 = value_l << exp
    move r1, 0     // r1 = 0
    jump r23


.globl count_all_ones_32
count_all_ones_32: // uint32_t count_all_ones_32(uint32_t value)
// Value is passed via r0 register
cao r0, r0
// Return from function. Return value in r0
jump r23 

.globl count_all_ones_64
count_all_ones_64: // uint32_t count_all_ones_64(uint64_t value)
    // Value is passed via r0 and r1 register
    cao r0, r0
    cao r1, r1
    add r0, r0, r1 // r0 = cao(r0) + cao(r1), fastadd because why are using even odd pair
    jump r23

.globl fast_and
fast_and: // uint64_t fast_and(uint64_t valueA, uint64_t valueB)
    // ValueA is passed via r0 and r1 register, Value B is passed via r2 and r3 register
    and r0, r0, r2 // r0 = r0 & r2
    and r1, r1, r3 // r1 = r1 & r3
    jump r23

.globl test_exponentiation
test_exponentiation: // uint64_t test_exp(uint64_t value, uint32_t exp)
 // value is passed via r0 (MBS), r1 (LBS) and r2 - exp

.globl fast_loop
fast_loop: // uint64_t fast_loop(const uint64_t *arr1, cosnt uint64_t *arr2)
#define PRECISION 32
#define arr1 r0
#define arr2 r1
#define exp r2
#define j r3
#define part_sum d4
#define tmp_addr r6
#define tmp_val r7
#define dot_product d8
#define val1 d10
#define val2 d12

    // arr1 pointer is in r0, arr2 pointer is in r1
    
    // dot product aka d8
    move r8, 0 // dot product value MSB
    move r9, 0 // dot product value LSB

    move exp, 62 // exp - (PRECISION - 1) << 1

// for (; exp >= PRECISION; exp--)
for_every_exp_top:
    // partsum is r4 and r5 aka d4
    move r4, 0 // partSum MSB
    move r5, 0 // partSum LSB

    add j, exp, -PRECISION + 1 // j = exp - PRECISION + 1 
    // for (j = exp - PRECISION + 1; j <= PRECISION -1; ++j)
    exp_top_inner:
        // Load val1 = arr1[j] - d10
        lsl_add tmp_addr, arr1, j, 3 // tmp_addr = arr1 + j<<3
        ld d10, tmp_addr, 0

        // Load val2 = arr2[exp - j] - d12
        sub tmp_val, exp, j // tmp_val = exp - j
        lsl_add tmp_addr, arr2, tmp_val, 3 // tmp_addr = arr2 + tmp_val<<3
        ld d12, tmp_addr, 0

        // val1 & val2 - "ANDed" number in d10
        and r10, r10, r12
        and r11, r11, r13

        // count all ones and store in r10
        cao r10, r10
        cao r11, r11
        add r10, r10, r11

        // partSum += r10 (all ones)
        // First add lower part
        add r5, r5, r10
        // NOT NEEDED: Add the carry to the MSB part
        // addc r4, r4, 0

        add j, j, 1 // j++

        // TODO: smarter jump 
        jgts j, PRECISION-1, exp_top_finish
        jump exp_top_inner
    exp_top_finish:

    // partSum = partSum << exp
    // We don't need to care about shifting top part, 
    // because max number of ones is 32*64, well within 32bits
    // lsl.u only works on bits 0:4 so max shift is 31, so we can 
    // treat r5 as it was the high part, shift it by exp-32 and then we add it to the MSB of dot product (r8) 
    add tmp_val, exp, -32
    lsl r5, r5, tmp_val // In r5 we now have partsum shifted by exp

    // dp += partSum
    // First lower part
    //add r9, r9, r5
    // Higher part
    add r8, r8, r5

    add exp, exp, -1 // exp = exp - 1 (exponent--)
    jlts exp, PRECISION, for_every_exp_bot // Jump to the next part, loop finished
    jump for_every_exp_top // Jump back, next iteration

// for (; exp >= 0; --exp)
for_every_exp_bot:
    // partsum is r4 and r5 aka d4
    move r4, 0 // partSum MSB
    move r5, 0 // partSum LSB
    
    move j, 0 // j = 0
    // for (j=0; j <= exp; j++)
    exp_bot_inner:
        // Load val1 = arr1[j] - d10
        lsl_add tmp_addr, arr1, j, 3 // tmp_addr = arr1 + j<<3
        ld d10, tmp_addr, 0

        // Load val2 = arr2[exp - j] - d12
        sub tmp_val, exp, j // tmp_val = exp - j
        lsl_add tmp_addr, arr2, tmp_val, 3 // tmp_addr = arr2 + tmp_val<<3
        ld d12, tmp_addr, 0

        // val1 & val2 - "ANDed" number in d10
        and r10, r10, r12
        and r11, r11, r13

        // count all ones and store in r10
        cao r10, r10
        cao r11, r11
        add r10, r10, r11

        // partSum += r10 (all ones)
        // First add lower part
        add r5, r5, r10
        // Not needed: Add the carry to the MSB part
        //addc r4, r4, 0

        add j, j, 1 // j++
        jgts j, exp, exp_bot_finish
        jump exp_bot_inner

    exp_bot_finish:
    
    // partSum = partSum << exp
    // We don't need to care about shifting top part, 
    // because max number of ones is 32*64, well within 32bits
    // lsl.u only works on bits 0:4 so max shift is 32, here it's ok,
    // because the exp is between 31 and 0
    lsl.u part_sum, r5, exp

    // dp += partSum
    // First lower part LSB
    add r9, r9, r5
    // Higher part MSB
    addc r8, r8, r4

    add exp, exp, -1 // exp = exp - 1
    jlts exp, 0, fast_loop_end
    jump for_every_exp_bot
    
fast_loop_end:
    // Set dot product as return value
    move r0, r8
    move r1, r9
    // Return from method
    jump r23
