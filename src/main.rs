use clap::Parser; // for argument parsing
mod parser;
mod patterns;
mod constraints;
// from parser.rs

/// Umiaq — pattern-matching word list solver
#[derive(Parser, Debug)]
#[command(author = "Alex Boisvert", version, about, long_about = None)]
struct Args {
    /// Pattern input string, e.g., "AB;BA;|A|=2;|B|=2;!=AB"
    input: String,

    /// Turn on debugging output
    #[arg(short, long)]
    debug: bool,

    /// Number of results to return
    #[arg(short = 'n', long, default_value_t = 100)]
    num_results: usize,

    /// Run test cases
    #[arg(short, long)]
    test: bool,
}

fn main() {
    let args = Args::parse();

    println!("Input: {}", args.input);
    println!("Debug: {}", args.debug);
    println!("Num results: {}", args.num_results);
    println!("Test mode: {}", args.test);
}
