// an attribute to hide warnings for unused code
#![allow(dead_code)]

/*
 * three types of structs:
 * 1: tuple structs: named tuples
 * 2: C structs
 * 3: unit structs: filed-less, useful for generics
 */

// we can use fmt::Debug or fmt::Display traits to print this struct,
// all types can derive the fmt::Debug implementation for printing, 
// but fmt::Display must be manually implemented
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
#[derive(Debug)]
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
    let saeid = Person {name, age};

    // print debug struct using :?, Debug trait is already implemented for many types,
    // to debug print a container, the items inside the container must also implement Debug trait.
    println!("{:?}", saeid);
    // pretty print
    println!("{:#?}", saeid);

    let point: Point = Point {x: 1.23, y: 4.56};
    println!("point coords: ({}, {})", point.x, point.y);
    
    // using struct update syntax to make use of coords of the other point struct
    let bottom_right = Point {x: 7.89, ..point};
    println!("bottom_right coords: ({}, {})", bottom_right.x, bottom_right.y);

    // destructure(?) the point using `let` binding
    let Point {x: left_edge, y: top_edge} = point;

    let rect = Rectangle {
        top_left: Point {x: left_edge, y: top_edge},
        bottom_right: bottom_right,
    };
    // error: 'Point' cannot be formatted with the default formatter, doesn't implement std::fmt::Display
   // println!("rect coords: ({}, {})", rect.top_left, rect.bottom_right);
   // print with debug formatter
   println!("rect coords: ({:?}, {:?})", rect.top_left, rect.bottom_right);
  
    let _unit = Unit;

    let pair = Pair(1, 3.14);
    println!("Tuple struct: ({}, {})", pair.0, pair.1);
    
    // destructure(?) a tuple struct
    let Pair(integer, decimal) = pair;
    println!("Pair struct: ({:?}, {:?})", integer, decimal);
}
