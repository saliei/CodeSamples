package main

import (
	"fmt"
	"math"
)

const s string = "a constant string"

func main() {
	fmt.Println(s)

	const n = 5000 // a const has no type unitl given one, e.g. by an explicit conversion
	fmt.Println(n)

	const e = 5e10 / n
	fmt.Println(e)

	fmt.Println(int64(n))

	fmt.Println(math.Cos(e)) // e here is given type float64, since Cos expects one
}
