import asyncio

import httpx

from research_lab.adapters import CrossrefAdapter, OpenAlexAdapter


async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        openalex = await OpenAlexAdapter(client).harvest("solid state battery electrolyte", 2)
        crossref = await CrossrefAdapter(client).harvest("solid state battery electrolyte", 2)
    print(
        {
            "openalex": len(openalex),
            "crossref": len(crossref),
            "openalex_dois": [record.doi for record in openalex],
            "crossref_dois": [record.doi for record in crossref],
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
