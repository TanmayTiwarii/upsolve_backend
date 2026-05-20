from typing import List
import httpx
import os

GRAPHQL_URL = os.getenv("GRAPHQL_URL", "https://leetcode.com/graphql/")

HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com",
}


async def fetch_all_question_ids(client: httpx.AsyncClient, slugs: List[str]) -> List[int]:
    """
    Fetch questionFrontendId for all given slugs in a SINGLE batched GraphQL query
    using field aliases instead of N separate requests.

    Example generated query:
        query {
            q0: question(titleSlug: "two-sum")     { questionFrontendId }
            q1: question(titleSlug: "rotate-image") { questionFrontendId }
        }
    """
    if not slugs:
        return []

    # Build aliased fields — one per slug
    fields = "\n".join(
        f'q{i}: question(titleSlug: "{slug}") {{ questionFrontendId }}'
        for i, slug in enumerate(slugs)
    )
    batched_query = {"query": f"query {{ {fields} }}"}

    try:
        resp = await client.post(GRAPHQL_URL, json=batched_query, headers=HEADERS)
        if resp.status_code != 200:
            print(f"Batched question query failed with status {resp.status_code}: {resp.text}")
            return []

        body = resp.json()
        if body.get("errors"):
            print(f"GraphQL errors in batched query: {body['errors']}")

        # data can be None if all aliases errored out — guard against that
        data = body.get("data") or {}
        ids = []
        for i in range(len(slugs)):
            question = data.get(f"q{i}")
            if question and question.get("questionFrontendId"):
                ids.append(int(question["questionFrontendId"]))
        return ids

    except Exception as e:
        print(f"Error during batched question fetch: {e}")
        return []


async def fetch_recent_problems(username: str) -> List[int]:
    query_recent = {
        "operationName": "recentAcSubmissions",
        "variables": {
            "username": username,
            "limit": 20
        },
        "query": "query recentAcSubmissions($username: String!, $limit: Int!) { recentAcSubmissionList(username: $username, limit: $limit) { id title titleSlug timestamp } }"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(GRAPHQL_URL, json=query_recent, headers=HEADERS)
            if resp.status_code != 200:
                print(f"Failed to fetch recent submissions for {username}: {resp.status_code} {resp.text}")
                return []

            body = resp.json()
            if body.get("errors"):
                print(f"GraphQL errors for recentAcSubmissions ({username}): {body['errors']}")

            submissions = (body.get("data") or {}).get("recentAcSubmissionList")
            if not submissions:
                print(f"No submissions found for user: {username}")
                return []

            # Extract unique titleSlugs
            title_slugs = list(set(
                sub.get("titleSlug") for sub in submissions if sub.get("titleSlug")
            ))

            # Single batched query instead of N individual requests
            return await fetch_all_question_ids(client, title_slugs)

    except httpx.ReadTimeout:
        print(f"Timeout when fetching recent problems for {username}")
        return []
    except Exception as e:
        print(f"Error fetching recent problems for {username}: {e}")
        return []
