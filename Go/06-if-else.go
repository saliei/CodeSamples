package main

import "fmt"

func main() {
	if 6%2 == 0 {
		fmt.Println("an even number")
	}

	if 10%3 == 0 {
		fmt.Println("divisible by 3")
	} else {
		fmt.Println("not divisible by 3")
	}

	if num := 15; num < 0 {
		fmt.Println("negative number")
	} else if num < 10 {
		fmt.Println("single digit num")
	} else {
		fmt.Println("multi-digit num")
	}

}
