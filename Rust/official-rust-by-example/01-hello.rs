fn main() {
    // printing is handled by a series of macros (identified by !) defined in std::fmt
    // format! : writes formatted text to String
    // print! : same as format!, prints to console (io::stdout)
    // println! : same as print!, with new line appended
    // eprint! : same print!, but to standard error (io::stderr)
    // eprintln! : same as eprint!, but new line appended

    println!("Hello world!");
    


    print!("{} days remaining!", 365);

    println!("I'm {0}, she's {1}. He is also {0}, and she is also {1}.", "Bob", "Alice");

    eprintln!("{func} {date} {subj}", func="main", date="17/11/22", subj="printing");

    println!("Base 10:               {}",   1024);
    println!("Base 2 (binary):       {:b}", 32768);
    println!("Base 8 (octal):        {:o}", 65);
    println!("Base 16 (hexadeciaml): {:x}", 1024);

    // right justify with a specified width:
    // total of 5 width right-justified
    println!("{num:>5}", num=4);
    println!("{num:>5}", num=32);
    // left-justify and zero pad
    println!("{num:0<5}", num=32);
    // append $ to use named args in format specifier
    println!("{num:>width$}", num=32, width=4);

    let num: f64 = 3.14;
    let width: usize = 5;
    println!("{num:>width$}");
    
    // pi with 4 floating point numbers
    let pi: f64 = 3.14159265;
    println!("{pi:.4}");


}
