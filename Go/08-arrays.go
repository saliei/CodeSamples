package main

import "fmt"

func main() {
	var a [5]int
	fmt.Println(a)

	a[1] = 10
	fmt.Println(a)

	fmt.Println("len: ", len(a))

	b := [3]string{"one", "two", "three"}
	fmt.Println("b[0]: ", b[0])

	var twoD [2][3]int
	twoD[1][2] = 2
	fmt.Println(twoD)

}
