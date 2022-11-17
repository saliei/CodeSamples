// an attribute to hide warnings for unused code
#![allow(dead_code)]

/*
 * three types of structs:
 * 1: tuple structs: named tuples
 * 2: C structs
 * 3: unit structs: filed-less, useful for generics
 */

#[derive(Debug)]
struct Person {
    name: String,
    age: u8,
}

// a unit struct
struct Unit;

// a tuple struct
struct Pair(i32, f32);

// a struct with two fields
struct Point {
    x: f32,
    y: f32,
}

// a struct can be reused as fields of another struct
struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}


fn main() {
    let name = String::from("Saeid");
    let age = 28;
    // create struct with field init shorthand
    let saeid = Person { name, age };

    println!("{:?}", saeid);
}
