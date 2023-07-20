package main

import (
	"fmt"
	"time"
)

func main() {
	i := 1
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
		fmt.Println("<12")
	case now.Hour() > 12:
		fmt.Println(">12")
	default:
		fmt.Println("noon")
	}

	var day = time.Now() // for top-level declarations, var is required // short hand declarations must be assigned

	switch day.Weekday() {
	case time.Saturday, time.Sunday:
		fmt.Println("weekend")
	default:
		fmt.Println("non-weekend``")
	}

	var j int
	fmt.Println(float64(j))
	fmt.Println(j)

	whatType := func(i interface{}) {
		switch t := i.(type) {
		case bool:
			fmt.Println("bool")
		case int:
			fmt.Println("int")
		default:
			fmt.Printf("type %T not handled", t)
		}
	}

	whatType(true)

}
