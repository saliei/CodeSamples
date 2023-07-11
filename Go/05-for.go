package main

import "fmt"

func main() {
	i := 1
	for i <= 5 {
		fmt.Println(i)
		i += 1
	}

	for j := 1; j < 5; j++ {
		fmt.Println(j)
	}

	for {
		fmt.Println("for without condition, breaked")
		break
	}

	for k := 2; k < 10; k++ {
		if k%2 == 0 {
			continue
		}
		fmt.Println(k)
	}
}
