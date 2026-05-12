def build_context(docs):
    context = ""
    for i, d in enumerate(docs):
        src = d.get("source", "Unknown")
        page = d.get("page", "Unknown")
        context += f"""
[Excerpt {i+1}]
Cite as: ({src}, p. {page})

{d['text']}
"""
    return context

def format_sources(docs):
    sources = []
    for i, d in enumerate(docs):
        sources.append(f"Source {i+1}: {d['source']} (page {d['page']})")
    return sources
