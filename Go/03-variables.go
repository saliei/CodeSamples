package main

import "fmt"

func main() {
	var s string = "initial"
	fmt.Println(s)

	var a, b int = 1, 2
	fmt.Println(a, b)

	var si = "string without type specified"
	fmt.Println(si)

	var d = true
	fmt.Println(d)

	var e int
	fmt.Println(e)

	f := "var declaration and initialization, only allowed inside a function"
	fmt.Println(f)

}
