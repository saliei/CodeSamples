	.file	"01-isqrt.cpp"
# GNU C++20 (GCC) version 13.1.1 20230429 (x86_64-pc-linux-gnu)
#	compiled by GNU C version 13.1.1 20230429, GMP version 6.2.1, MPFR version 4.2.0, MPC version 1.3.1, isl version isl-0.26-GMP

# warning: MPFR header version 4.2.0 differs from library version 4.2.0-p7.
# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mtune=generic -march=x86-64 -O2 -std=c++20
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%d\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB15:
	.cfi_startproc
	subq	$8, %rsp	#,
	.cfi_def_cfa_offset 16
# 01-isqrt.cpp:11:     printf("%d\n", s);
	movl	$42, %esi	#,
	leaq	.LC0(%rip), %rdi	#, tmp83
	xorl	%eax, %eax	#
	call	printf@PLT	#
# 01-isqrt.cpp:14: }
	xorl	%eax, %eax	#
	addq	$8, %rsp	#,
	.cfi_def_cfa_offset 8
	ret	
	.cfi_endproc
.LFE15:
	.size	main, .-main
	.ident	"GCC: (GNU) 13.1.1 20230429"
	.section	.note.GNU-stack,"",@progbits
