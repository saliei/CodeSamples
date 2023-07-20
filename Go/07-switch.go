package main

import (
	"fmt"
	"time"
)

func main() {
	i := 2
	fmt.Print("write ", i, " as: ")
	switch i {
	case 1:
		fmt.Println("one")
	case 2:
		fmt.Println("two")
	case 3:
		fmt.Println("three")
	}

	now := time.Now()
	switch {
	case now.Hour() < 12:
		fmt.Println("hour is < 12")
	default:
		fmt.Println("hour is > 12")
	}

	day := time.Now()
	switch day.Weekday() {
	case time.Saturday, time.Sunday:
		fmt.Println("it's weekend!")
	default:
		fmt.Println("not weekened")
	}

	whatType := func(i interface{}) { // an empty interface may hold values of any type
		switch t := i.(type) {
		case bool:
			fmt.Println("bool")
		case int:
			fmt.Println("int")
		default:
			fmt.Printf("Type: %T is not handled\n", t)
		}
	}

	whatType(true)
	whatType(10)
	whatType(int64(20))
	whatType("hi")
}
