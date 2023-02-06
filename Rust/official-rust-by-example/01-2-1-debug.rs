// not printable by either `fmt::Display` or `fmt::Debug` trait
struct UnPrintable(i32);

// automatically `fmt::Debug` printable by derive attribute
#[derive(Debug)]
struct DebugPrintable(i32);

#[derive(Debug)]
struct Structure(i32);

#[derive(Debug)]
struct Deep(Structure);

#[derive(Debug)]
struct Person1{
    name: str,
    age: u8
}

#[derive(Debug)]
struct Person2<'a> {
    name: &'a str,
    age: u8
}



fn main() {
    // all `std` library types are automatically {:?} printable
    println!("{:?} hours in a day.", 24);
    println!("{1:?} is better than {0:?}, says {person:?}", "emacs", "vim", person="no one");

    println!("Structure Debug print: {:?}", Structure(1));
    println!("Deep Structure Debug print: {:?}", Deep(Structure(2)));

    // pretty printing
    println!("Structure Debug pretty print: {:#?}", Structure(1));
    println!("Deep Structure Debug pretty print: {:#?}", Deep(Structure(2)));

    let name = "Samuel";
    let age = 30;
    let sam1 = Person1 {name, age};
    let sam2 = Person2 {name, age};
        

    println!("Person1 pretty print: {:#?}", sam1);
    println!("Person2 pretty print: {:#?}", sam2);

}
