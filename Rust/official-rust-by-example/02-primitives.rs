// scalar types:
// int: i8, i16, i32 (default), i64, i128 and isize(? pointer size)
// unsigned int: u8, u16, u32, u64, u128, and usize (? pointer size)
// float: f32, f64 (default)
// char: unicode scalar values: 'a', 'α', '∞' (4 bytes)
// bool: true, false
// unit type: (): whose only possible value is an empty tuple: (), ?
// compound types:
// arrays like: [1, 2, 3]
// tuples like: (1, true)

fn main() {
    // unused variable prefixed with underscore to suppress compiler warning
    let _logical: bool = true;

    let a_float: f64 = 1.0; // regular annotation
    let an_integer = 5i32; // suffix annotation
    
    // with no annotation, default will be used
    let default_float = 3.14; // f64
    let default_integer = 7; // i32
                            

    // a type can also be inferred from context
    // this will also spit a warning: value assigned to inferred_type is never used,
    // because we immediately overwrite it and 12 is never used!
    let mut inferred_type = 12; // i64 type is inferred from the next line
    inferred_type = 4294967296i64;

    let mut mutable = 12; // a mutable i32
    println!("mutable         = {}", mutable);
    mutable = 21;
    println!("mutable         = {}", mutable);

    // _mutable = true; // Error! expected integer, found 'bool', the type of a variable can't be changed!
    // variables can be overwritten with shadowing
    let mutable = true;

    println!("a_float         = {}", a_float);
    println!("an_integer      = {}", an_integer);
    println!("default_integer = {}", default_integer);
    println!("default_float   = {}", default_float);
    println!("inferred_type   = {}", inferred_type);
    println!("mutable         = {}", mutable);

}
