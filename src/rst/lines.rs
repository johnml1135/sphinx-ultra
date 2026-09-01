//! Line preprocessing: docutils `statemachine.string2lines(tab_width=8,
//! convert_whitespace=True)` equivalent, producing the parser's line stream
//! as index-based [`LineRec`] records over a processed source text.
//!
//! Probe-verified semantics (docutils 0.22.4):
//! - `\v` / `\f` become single spaces (convert_whitespace).
//! - Lines split on `\n`, `\r\n`, `\r` (plus Python `splitlines` exotics:
//!   `\x1c`-`\x1e`, `\u{85}`, `\u{2028}`, `\u{2029}`).
//! - Tabs expand to the next multiple-of-8 column (character columns).
//! - Trailing whitespace stripped per line.
//! - One processed line == one source line, so message line numbers are
//!   `processed index + 1`.

/// One line of the parser's stream: `(source, lineno)` provenance plus the
/// byte range of the line's current (possibly dedented) view into the
/// *processed* text of its source.
///
/// `source` indexes the parser's source-text table (entry 0 is the document
/// itself; included files push further entries). Holding indices rather
/// than borrows is the load-bearing choice: the table can grow mid-parse
/// (an included file splices its lines into the running stream) without
/// any record referencing parser-owned text, so no self-reference and no
/// unsafe code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRec {
    /// Index into the parser's source table (0 = the document itself).
    pub source: u16,
    /// 1-based line number within `source`.
    pub lineno: u32,
    /// Byte range of the current view into the processed text of `source`.
    pub start: u32,
    pub end: u32,
    /// Cached leading-space count. Computed once at record construction and
    /// derived arithmetically on dedent — re-scanning per nesting level made
    /// deep nesting O(depth^3) (measured: 800-level nest took ~0.5s).
    indent: u32,
}

impl LineRec {
    /// Build a record over `line_text`, which must be exactly the
    /// `start..end` slice of the source's processed text.
    pub(crate) fn new(source: u16, lineno: u32, start: u32, end: u32, line_text: &str) -> LineRec {
        debug_assert_eq!(line_text.len(), (end - start) as usize);
        let indent = (line_text.len() - line_text.trim_start_matches(' ').len()) as u32;
        LineRec {
            source,
            lineno,
            start,
            end,
            indent,
        }
    }

    pub(crate) fn is_blank(&self) -> bool {
        self.start == self.end
    }

    pub(crate) fn indent(&self) -> usize {
        self.indent as usize
    }

    /// Dedent by `n` columns (leading columns are spaces by construction;
    /// marker lines are re-wrapped with [`LineRec::new`] instead).
    pub(crate) fn dedented(&self, n: usize) -> LineRec {
        let n = n.min(self.indent()) as u32;
        LineRec {
            start: self.start + n,
            indent: self.indent - n,
            ..*self
        }
    }

    /// The line's current view within its source's processed `text`.
    pub(crate) fn slice<'t>(&self, text: &'t str) -> &'t str {
        &text[self.start as usize..self.end as usize]
    }
}

/// The processed form of one source: its lines joined with `\n` (the text
/// every [`LineRec`] range indexes) plus the single-source record stream.
#[derive(Debug, Clone, Default)]
pub struct Lines {
    text: String,
    recs: Vec<LineRec>,
}

fn is_line_boundary(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

fn process_line(raw: &str, out: &mut String) {
    // convert_whitespace (\v, \f -> space), then expandtabs(8), then rstrip.
    let base = out.len();
    let mut col = 0usize;
    for c in raw.chars() {
        match c {
            '\t' => {
                let next_stop = (col / 8 + 1) * 8;
                for _ in col..next_stop {
                    out.push(' ');
                }
                col = next_stop;
            }
            '\x0b' | '\x0c' => {
                out.push(' ');
                col += 1;
            }
            _ => {
                out.push(c);
                col += 1;
            }
        }
    }
    out.truncate(base + out[base..].trim_end().len());
}

impl Lines {
    /// Process `source` into the single-source form: source id 0, linenos
    /// `1..=n`.
    pub fn new(source: &str) -> Lines {
        Lines::for_source(source, 0, 1)
    }

    /// Process `source` as the table entry `source_id`, numbering its lines
    /// from `first_lineno` (an included file numbers from 1; a sub-parse of
    /// text lifted out of a document keeps the document's numbering).
    pub(crate) fn for_source(source: &str, source_id: u16, first_lineno: u32) -> Lines {
        let mut text = String::with_capacity(source.len());
        let mut recs = Vec::new();
        let push_line = |raw: &str, text: &mut String, recs: &mut Vec<LineRec>| {
            if !recs.is_empty() {
                text.push('\n');
            }
            let start = text.len() as u32;
            process_line(raw, text);
            let end = text.len() as u32;
            let lineno = first_lineno + recs.len() as u32;
            recs.push(LineRec::new(
                source_id,
                lineno,
                start,
                end,
                &text[start as usize..end as usize],
            ));
        };
        let bytes_len = source.len();
        let mut line_start = 0usize;
        let mut chars = source.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if is_line_boundary(c) {
                push_line(&source[line_start..i], &mut text, &mut recs);
                // \r\n is one boundary
                if c == '\r' {
                    if let Some(&(_, '\n')) = chars.peek() {
                        chars.next();
                    }
                }
                line_start = match chars.peek() {
                    Some(&(j, _)) => j,
                    None => bytes_len,
                };
            }
        }
        if line_start < bytes_len {
            push_line(&source[line_start..], &mut text, &mut recs);
        }
        Lines { text, recs }
    }

    #[cfg(test)]
    pub(crate) fn recs(&self) -> &[LineRec] {
        &self.recs
    }

    /// The processed text and the record stream, for a parser taking
    /// ownership of both.
    pub(crate) fn into_parts(self) -> (String, Vec<LineRec>) {
        (self.text, self.recs)
    }

    #[cfg(test)]
    fn line_str(&self, i: usize) -> &str {
        self.recs[i].slice(&self.text)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.recs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabs_expand_to_8_col_stops() {
        assert_eq!(Lines::new("a\tb").line_str(0), "a       b");
        assert_eq!(Lines::new("\ta").line_str(0), "        a");
        assert_eq!(Lines::new("ab\tc").line_str(0), "ab      c");
        assert_eq!(Lines::new("abcdefgh\tz").line_str(0), "abcdefgh        z");
        assert_eq!(Lines::new("x\ty\tz").line_str(0), "x       y       z");
    }

    #[test]
    fn trailing_whitespace_stripped_and_recs_map_into_processed_text() {
        let l = Lines::new("one  \ntwo");
        assert_eq!(l.line_str(0), "one");
        assert_eq!(l.line_str(1), "two");
        // Ranges index the PROCESSED text (trailing whitespace gone).
        assert_eq!(l.text, "one\ntwo");
        assert_eq!(
            (l.recs[1].start as usize, l.recs[1].end as usize),
            (4, 7),
            "second line's range starts past the first line and its separator"
        );
    }

    #[test]
    fn a_three_line_document_yields_source_0_linenos_1_to_3() {
        let l = Lines::new("one\ntwo\nthree");
        let stream: Vec<(u16, u32)> = l.recs().iter().map(|r| (r.source, r.lineno)).collect();
        assert_eq!(stream, vec![(0, 1), (0, 2), (0, 3)]);
    }

    #[test]
    fn for_source_stamps_the_given_source_id_and_base_lineno() {
        let l = Lines::for_source("a\nb", 3, 10);
        let stream: Vec<(u16, u32)> = l.recs().iter().map(|r| (r.source, r.lineno)).collect();
        assert_eq!(stream, vec![(3, 10), (3, 11)]);
    }

    #[test]
    fn vertical_tab_and_formfeed_become_spaces() {
        assert_eq!(Lines::new("a\x0bb\x0cc").line_str(0), "a b c");
    }

    #[test]
    fn crlf_and_cr_split_without_stray_cr() {
        let l = Lines::new("one\r\ntwo\rthree");
        assert_eq!(l.len(), 3);
        assert_eq!(l.line_str(0), "one");
        assert_eq!(l.line_str(1), "two");
        assert_eq!(l.line_str(2), "three");
    }

    #[test]
    fn trailing_newline_produces_no_empty_last_line() {
        assert_eq!(Lines::new("a\n").len(), 1);
        let l = Lines::new("a\n\n");
        assert_eq!(l.len(), 2);
        assert!(l.recs[1].is_blank());
    }

    #[test]
    fn indent_counts_leading_spaces() {
        let l = Lines::new("    four\n\tone-tab");
        assert_eq!(l.recs[0].indent(), 4);
        assert_eq!(l.recs[1].indent(), 8);
    }

    #[test]
    fn dedent_moves_the_view_start_and_shrinks_the_cached_indent() {
        let l = Lines::new("    body");
        let d = l.recs[0].dedented(4);
        assert_eq!(d.slice(&l.text), "body");
        assert_eq!(d.indent(), 0);
        assert_eq!(d.lineno, 1);
    }
}
