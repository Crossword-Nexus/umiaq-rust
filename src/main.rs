use clap::Parser;
use std::time::Instant;

use umiaq_rust::bindings::Bindings;
use umiaq_rust::wordlist;
use umiaq_rust::solver;

/// Umiaq pattern solver
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The pattern to solve (e.g. "AB;BA;|A|=2;|B|=2;!=AB")
    pattern: String,

    /// Path to the word list file (word;score per line)
    #[arg(
        short,
        long,
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/spreadthewordlist.dict")
    )]
    wordlist: String,

    /// Minimum score filter
    #[arg(short = 'm', long, default_value_t = 50)]
    min_score: i32,

    /// Maximum allowed word length
    #[arg(short = 'L', long, default_value_t = 21)]
    max_len: usize,

    /// Maximum number of results to return
    #[arg(short = 'n', long, default_value_t = 100)]
    num_results_requested: usize,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let t_load = Instant::now();
    let wl = wordlist::WordList::load_from_path(&cli.wordlist, cli.min_score, cli.max_len)?;
    let load_secs = t_load.elapsed().as_secs_f64();

    let words_ref: Vec<&str> = wl.entries.iter().map(String::as_str).collect();

    let t_solve = Instant::now();
    let solutions: Vec<Vec<Bindings>> = solver::solve_equation(&cli.pattern, &words_ref, cli.num_results_requested)?;
    let solve_secs = t_solve.elapsed().as_secs_f64();

    for solution in &solutions {
        // We display the results in uppercase and separated with a bullet
        let display = solution.iter()
            .map(|b| b.get_word().unwrap().to_ascii_uppercase())
            .collect::<Vec<_>>()
            .join(" • ");
        println!("{display}");
    }

    eprintln!(
        "Loaded {} words in {:.3}s; solved in {:.3}s ({} tuples).",
        wl.entries.len(), load_secs, solve_secs, solutions.len()
    );

    Ok(())
}
