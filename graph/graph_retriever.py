from neo4j import GraphDatabase
from config import settings

class GraphRetriever:

    def __init__(self, workspace):
        self.workspace = workspace
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USERNAME,
                settings.NEO4J_PASSWORD
            )
        )

    def retrieve(self, chunk_ids):

        related_chunks = set()
        related_concepts = set()
        relationships = []

        with self.driver.session() as session:
            for chunk_id in chunk_ids:
                related_chunks.update(
                    self.get_related_chunks(
                        session,
                        chunk_id
                    )
                )
                related_concepts.update(
                    self.get_related_concepts(
                        session,
                        chunk_id
                    )
                )
                relationships.extend(
                    self.get_relationships(
                        session,
                        chunk_id
                    )
                )

        return {
            "related_chunk_ids": list(related_chunks),
            "related_concepts": sorted(related_concepts),
            "relationships": self.remove_duplicate_relationships(relationships)
        }

    def get_related_chunks(
        self,
        session,
        chunk_id
    ):

        query = """
        MATCH (c:Chunk {
            id:$chunk,
            workspace:$workspace
        })
            -[:MENTIONS]->
            (concept:Concept {
                workspace:$workspace
            })
        MATCH (concept)-[*1..2]-(related:Concept {
            workspace:$workspace
        })
        MATCH (other:Chunk {
            workspace:$workspace
        })
            -[:MENTIONS]->
            (related)
        WHERE other.id <> $chunk
        RETURN DISTINCT other.id AS chunk
        """

        result = session.run(
            query,
            chunk=chunk_id,
            workspace=self.workspace
        )

        return [
            row["chunk"]
            for row in result
        ]

    def get_related_concepts(
        self,
        session,
        chunk_id
    ):

        query = """
        MATCH (c:Chunk {
            id:$chunk,
            workspace:$workspace
        })
            -[:MENTIONS]->
            (concept:Concept {
                workspace:$workspace
            })
        OPTIONAL MATCH
            (concept)-[*1..2]-(related:Concept {
                workspace:$workspace
            })
        RETURN DISTINCT
            concept.name AS concept,
            related.name AS related
        """

        result = session.run(
            query,
            chunk=chunk_id,
            workspace=self.workspace
        )

        concepts = set()

        for row in result:
            if row["concept"]:
                concepts.add(row["concept"])
            if row["related"]:
                concepts.add(row["related"])

        return list(concepts)

    def get_relationships(
        self,
        session,
        chunk_id
    ):

        query = """
        MATCH (c:Chunk {
            id:$chunk,
            workspace:$workspace
        })
            -[:MENTIONS]->
            (a:Concept {
                workspace:$workspace
            })
        MATCH (a)-[r]->(b:Concept {
            workspace:$workspace
        })
        RETURN DISTINCT
            a.name AS source,
            type(r) AS relationship,
            b.name AS target
        """

        result = session.run(
            query,
            chunk=chunk_id,
            workspace=self.workspace
        )

        return [
            {
                "source": row["source"],
                "relationship": row["relationship"],
                "target": row["target"]
            }
            for row in result
        ]

    def remove_duplicate_relationships(
        self,
        relationships
    ):

        unique = {}
        for relation in relationships:
            key = (
                relation["source"],
                relation["relationship"],
                relation["target"]
            )
            unique[key] = relation

        return list(unique.values())

    def clear(self):

        with self.driver.session() as session:
            session.run(
                """
                MATCH (n)
                WHERE n.workspace=$workspace
                DETACH DELETE n
                """,
                workspace=self.workspace
                )

    def close(self):
        self.driver.close()