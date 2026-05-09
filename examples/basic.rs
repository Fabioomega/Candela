use candela::arange;
use tracing::Level;

fn main() {
    tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_max_level(Level::DEBUG)
        .init();

    let t = arange!(7 * 4).view(&[7, 4]).unwrap();
    let t2 = arange!(7 * 4).view(&[1, 7, 4]).unwrap();
    let t3 = t + t2;
}
