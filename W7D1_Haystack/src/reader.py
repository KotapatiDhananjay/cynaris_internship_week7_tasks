
from haystack import component


@component
class Reader:
    """Simple Reader component for retrieved document excerpts."""

    @component.output_types(results=list[dict])
    def run(self, documents: list):

        results = []

        for document in documents:

            results.append({
                "source": document.meta.get(
                    "file_path",
                    document.meta.get("file_name", "Unknown")
                ),
                "score": document.score,
                "answer": document.content[:500]
            })

        return {"results": results}
