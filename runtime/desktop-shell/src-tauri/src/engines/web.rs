// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Read;
use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;
use url::{Host, Url};

const SEARCH_ENDPOINT: &str = "https://html.duckduckgo.com/html/";
const USER_AGENT: &str = "QuantoniumOS/0.1 (secure-web-tool)";
const MAX_QUERY_LEN: usize = 220;
const MAX_RESULTS: usize = 5;
const MAX_BODY_BYTES: usize = 256 * 1024;
const MAX_FETCH_CHARS: usize = 1600;
const MAX_CONTEXT_CHARS: usize = 8_000;
const MAX_FILTER_DOMAINS: usize = 6;

#[derive(Debug, Clone, Deserialize)]
pub struct WebLookupRequest {
    pub query: String,
    #[serde(default)]
    pub domains: Option<Vec<String>>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebFetchRequest {
    pub url: String,
    #[serde(default)]
    pub max_chars: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WebSource {
    pub title: String,
    pub url: String,
    pub domain: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WebLookupResult {
    pub success: bool,
    pub query: String,
    pub sources: Vec<WebSource>,
    pub context: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WebFetchResult {
    pub success: bool,
    pub title: String,
    pub url: String,
    pub domain: String,
    pub excerpt: String,
    pub context: String,
    pub error: String,
}

pub fn web_lookup(req: WebLookupRequest) -> Result<WebLookupResult, String> {
    let query = sanitize_query(&req.query)?;
    let domain_filters = sanitize_domain_filters(req.domains.unwrap_or_default())?;
    let limit = req.limit.unwrap_or(4).clamp(1, MAX_RESULTS);

    let search_url = build_search_url(&query)?;
    let html = fetch_text_response(&search_url, true)?;
    let sources = extract_search_results(&html, &domain_filters, limit)?;
    if sources.is_empty() {
        return Ok(WebLookupResult {
            success: false,
            query,
            sources: Vec::new(),
            context: String::new(),
            error: "No safe public web results were found for that query.".to_string(),
        });
    }

    let mut context_parts = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        context_parts.push(format!(
            "[{}] {} ({})\nURL: {}\nSearch snippet: {}",
            index + 1,
            source.title,
            source.domain,
            source.url,
            source.snippet
        ));
    }

    for (index, source) in sources.iter().take(2).enumerate() {
        if let Ok(fetch_result) = web_fetch(WebFetchRequest {
            url: source.url.clone(),
            max_chars: Some(900),
        }) {
            if fetch_result.success && !fetch_result.excerpt.is_empty() {
                context_parts.push(format!(
                    "[{}] Page excerpt from {}:\n{}",
                    index + 1,
                    fetch_result.domain,
                    fetch_result.excerpt
                ));
            }
        }
    }

    let context = clamp_chars(&context_parts.join("\n\n"), MAX_CONTEXT_CHARS);

    Ok(WebLookupResult {
        success: true,
        query,
        sources,
        context,
        error: String::new(),
    })
}

pub fn web_fetch(req: WebFetchRequest) -> Result<WebFetchResult, String> {
    let max_chars = req.max_chars.unwrap_or(900).clamp(200, MAX_FETCH_CHARS);
    let url = validate_public_https_url(req.url.trim())?;
    let html = fetch_text_response(url.as_str(), false)?;
    let title = extract_title(&html).unwrap_or_else(|| url.host_str().unwrap_or("web page").to_string());
    let excerpt = extract_text_excerpt(&html, max_chars);
    let domain = domain_from_url(&url);
    let context = format!(
        "{} ({})\nURL: {}\nExcerpt: {}",
        title,
        domain,
        url,
        excerpt
    );

    Ok(WebFetchResult {
        success: true,
        title,
        url: url.to_string(),
        domain,
        excerpt,
        context,
        error: String::new(),
    })
}

fn build_search_url(query: &str) -> Result<String, String> {
    let mut url = Url::parse(SEARCH_ENDPOINT).map_err(|err| format!("Search URL parse failed: {err}"))?;
    url.query_pairs_mut().append_pair("q", query);
    Ok(url.to_string())
}

fn fetch_text_response(url: &str, allow_search_endpoint: bool) -> Result<String, String> {
    if !allow_search_endpoint {
        let _ = validate_public_https_url(url)?;
    }

    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(10))
        .timeout_write(Duration::from_secs(10))
        .build();

    let response = agent
        .get(url)
        .set("User-Agent", USER_AGENT)
        .set("Accept", "text/html,application/xhtml+xml")
        .timeout(Duration::from_secs(12))
        .call()
        .map_err(|err| format!("Web request failed: {err}"))?;

    let content_type = response
        .header("Content-Type")
        .unwrap_or("")
        .to_ascii_lowercase();
    if !content_type.is_empty()
        && !content_type.contains("text/html")
        && !content_type.contains("application/xhtml+xml")
    {
        return Err(format!("Unsupported content type: {content_type}"));
    }

    read_limited_body(response, MAX_BODY_BYTES)
}

fn read_limited_body(response: ureq::Response, limit: usize) -> Result<String, String> {
    let mut reader = response.into_reader().take((limit + 1) as u64);
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .map_err(|err| format!("Read failed: {err}"))?;
    if buffer.len() > limit {
        return Err("Response exceeded the secure size limit.".to_string());
    }
    Ok(String::from_utf8_lossy(&buffer).to_string())
}

fn sanitize_query(raw: &str) -> Result<String, String> {
    let collapsed = collapse_whitespace(raw);
    if collapsed.is_empty() {
        return Err("Web lookup query is empty.".to_string());
    }
    Ok(clamp_chars(&collapsed, MAX_QUERY_LEN))
}

fn sanitize_domain_filters(domains: Vec<String>) -> Result<Vec<String>, String> {
    let domain_re = Regex::new(r"(?i)^[a-z0-9][a-z0-9.-]*[a-z0-9]$").unwrap();
    let mut clean = Vec::new();

    for raw in domains.into_iter().take(MAX_FILTER_DOMAINS) {
        let domain = raw.trim().trim_matches('.').to_ascii_lowercase();
        if domain.is_empty() {
            continue;
        }
        if !domain_re.is_match(&domain) {
            return Err(format!("Invalid domain filter: {domain}"));
        }
        if is_blocked_domain_suffix(&domain) {
            return Err(format!("Blocked private domain filter: {domain}"));
        }
        clean.push(domain);
    }

    Ok(clean)
}

fn extract_search_results(html: &str, domain_filters: &[String], limit: usize) -> Result<Vec<WebSource>, String> {
    let link_re = Regex::new(r#"(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#)
        .map_err(|err| format!("Search parser init failed: {err}"))?;
    let snippet_re = Regex::new(
        r#"(?is)<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>"#,
    )
    .map_err(|err| format!("Snippet parser init failed: {err}"))?;

    let snippets: Vec<String> = snippet_re
        .captures_iter(html)
        .filter_map(|capture| capture.get(1).map(|m| clean_html_fragment(m.as_str())))
        .collect();

    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for (index, capture) in link_re.captures_iter(html).enumerate() {
        let raw_href = capture.get(1).map(|m| decode_html_entities(m.as_str())).unwrap_or_default();
        let title = capture.get(2).map(|m| clean_html_fragment(m.as_str())).unwrap_or_default();
        if title.is_empty() {
            continue;
        }

        let resolved_url = match resolve_result_url(&raw_href) {
            Ok(url) => url,
            Err(_) => continue,
        };
        let validated_url = match validate_public_https_url(resolved_url.as_str()) {
            Ok(url) => url,
            Err(_) => continue,
        };
        let domain = domain_from_url(&validated_url);
        if !domain_matches_filters(&domain, domain_filters) {
            continue;
        }
        let url_string = validated_url.to_string();
        if !seen.insert(url_string.clone()) {
            continue;
        }

        results.push(WebSource {
            title,
            url: url_string,
            domain,
            snippet: snippets.get(index).cloned().unwrap_or_default(),
        });

        if results.len() >= limit {
            break;
        }
    }

    Ok(results)
}

fn resolve_result_url(raw_href: &str) -> Result<Url, String> {
    let normalized = if raw_href.starts_with("//") {
        format!("https:{raw_href}")
    } else {
        raw_href.to_string()
    };
    let parsed = Url::parse(&normalized)
        .or_else(|_| Url::parse(&format!("https://duckduckgo.com{normalized}")))
        .map_err(|err| format!("Invalid result URL: {err}"))?;

    if parsed.domain() == Some("duckduckgo.com") {
        for (key, value) in parsed.query_pairs() {
            if key == "uddg" {
                return Url::parse(value.as_ref())
                    .map_err(|err| format!("Invalid redirected result URL: {err}"));
            }
        }
    }

    Ok(parsed)
}

fn validate_public_https_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw).map_err(|err| format!("Invalid URL: {err}"))?;
    if url.scheme() != "https" {
        return Err("Only https URLs are allowed for secure web access.".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("Credential-bearing URLs are blocked.".to_string());
    }
    if let Some(port) = url.port() {
        if port != 443 {
            return Err("Non-standard ports are blocked.".to_string());
        }
    }

    match url.host() {
        Some(Host::Domain(domain)) => {
            let domain = domain.to_ascii_lowercase();
            if is_blocked_domain_suffix(&domain) {
                return Err(format!("Blocked private or local host: {domain}"));
            }
            let mut resolved_any = false;
            for address in (domain.as_str(), 443)
                .to_socket_addrs()
                .map_err(|err| format!("DNS resolution failed: {err}"))?
            {
                resolved_any = true;
                if !is_public_ip(address.ip()) {
                    return Err(format!("Blocked private network target: {}", address.ip()));
                }
            }
            if !resolved_any {
                return Err("Host did not resolve to a public address.".to_string());
            }
        }
        Some(Host::Ipv4(ip)) => {
            if !is_public_ip(IpAddr::V4(ip)) {
                return Err(format!("Blocked private network target: {ip}"));
            }
        }
        Some(Host::Ipv6(ip)) => {
            if !is_public_ip(IpAddr::V6(ip)) {
                return Err(format!("Blocked private network target: {ip}"));
            }
        }
        None => return Err("URL must include a host.".to_string()),
    }

    Ok(url)
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            !ip.is_private()
                && !ip.is_loopback()
                && !ip.is_link_local()
                && !ip.is_multicast()
                && !ip.is_broadcast()
                && !ip.is_documentation()
                && !ip.is_unspecified()
                && ip.octets()[0] != 0
                && !(ip.octets()[0] == 100 && (64..=127).contains(&ip.octets()[1]))
                && !(ip.octets()[0] == 198 && ip.octets()[1] == 18)
                && !(ip.octets()[0] == 198 && ip.octets()[1] == 19)
        }
        IpAddr::V6(ip) => {
            !ip.is_loopback()
                && !ip.is_unspecified()
                && !ip.is_multicast()
                && !ip.is_unique_local()
                && !ip.is_unicast_link_local()
                && !ip.segments().starts_with(&[0x2001, 0x0db8])
        }
    }
}

fn is_blocked_domain_suffix(domain: &str) -> bool {
    domain == "localhost"
        || domain.ends_with(".localhost")
        || domain.ends_with(".local")
        || domain.ends_with(".internal")
        || domain.ends_with(".lan")
        || domain.ends_with(".home")
        || domain.ends_with(".arpa")
}

fn domain_matches_filters(domain: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    filters.iter().any(|filter| domain == filter || domain.ends_with(&format!(".{filter}")))
}

fn domain_from_url(url: &Url) -> String {
    url.host_str().unwrap_or("unknown").to_ascii_lowercase()
}

fn extract_title(html: &str) -> Option<String> {
    let title_re = Regex::new(r#"(?is)<title[^>]*>(.*?)</title>"#).ok()?;
    title_re
        .captures(html)
        .and_then(|capture| capture.get(1))
        .map(|m| clean_html_fragment(m.as_str()))
        .filter(|text| !text.is_empty())
}

fn extract_text_excerpt(html: &str, max_chars: usize) -> String {
    let stripped = strip_non_text_blocks(html);
    let text = strip_tags(&stripped);
    clamp_chars(&collapse_whitespace(&decode_html_entities(&text)), max_chars)
}

fn strip_non_text_blocks(html: &str) -> String {
    let patterns = [
        r"(?is)<script[^>]*>.*?</script>",
        r"(?is)<style[^>]*>.*?</style>",
        r"(?is)<noscript[^>]*>.*?</noscript>",
        r"(?is)<svg[^>]*>.*?</svg>",
        r"(?is)<iframe[^>]*>.*?</iframe>",
        r"(?is)<object[^>]*>.*?</object>",
        r"(?is)<template[^>]*>.*?</template>",
    ];

    let mut cleaned = html.to_string();
    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            cleaned = re.replace_all(&cleaned, " ").to_string();
        }
    }
    cleaned
}

fn strip_tags(input: &str) -> String {
    Regex::new(r"(?is)<[^>]+>")
        .map(|re| re.replace_all(input, " ").to_string())
        .unwrap_or_else(|_| input.to_string())
}

fn clean_html_fragment(input: &str) -> String {
    collapse_whitespace(&decode_html_entities(&strip_tags(input)))
}

fn collapse_whitespace(input: &str) -> String {
    input
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn clamp_chars(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect::<String>()
}

fn decode_html_entities(input: &str) -> String {
    let mut output = input
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#x2F;", "/");

    if let Ok(re) = Regex::new(r"&#(\d+);") {
        output = re
            .replace_all(&output, |captures: &regex::Captures| {
                captures
                    .get(1)
                    .and_then(|m| m.as_str().parse::<u32>().ok())
                    .and_then(char::from_u32)
                    .map(|c| c.to_string())
                    .unwrap_or_default()
            })
            .to_string();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_private_urls() {
        assert!(validate_public_https_url("https://127.0.0.1/admin").is_err());
        assert!(validate_public_https_url("https://192.168.1.1").is_err());
        assert!(validate_public_https_url("https://localhost/test").is_err());
        assert!(validate_public_https_url("http://example.com").is_err());
    }

    #[test]
    fn cleans_html_excerpt() {
        let html = r#"
            <html>
              <head><title>Example &amp; Test</title><style>body{}</style></head>
              <body><script>alert(1)</script><h1>Hello</h1><p>World&nbsp;now</p></body>
            </html>
        "#;
        assert_eq!(extract_title(html).as_deref(), Some("Example & Test"));
        assert_eq!(extract_text_excerpt(html, 200), "Example & Test Hello World now");
    }

    #[test]
    fn resolves_duckduckgo_redirect_url() {
        let redirected = resolve_result_url(
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle",
        )
        .expect("redirect URL should parse");
        assert_eq!(redirected.as_str(), "https://example.com/article");
    }
}
