import json
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from config import settings

class GraphBuilder:

    def __init__(self, workspace):

        self.workspace=workspace
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(
                settings.NEO4J_USERNAME,
                settings.NEO4J_PASSWORD
            )
        )

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0
        )

    def build(self, chunks):

        with self.driver.session() as session:
            for chunk in chunks:
                self.create_chunk_node(session, chunk)
                concepts, relationships = self.extract_graph(chunk.text)
                print("Concepts:", concepts)
                print("Relationships:", relationships)
                self.create_concepts(
                    session,
                    chunk.id,
                    concepts
                )
                self.create_relationships(
                    session,
                    relationships
                )

    def create_chunk_node(self, session, chunk):

        session.run(
            """
            MERGE (d:Document {name:$source})
            MERGE (c:Chunk {id:$id})
            SET c.page=$page
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            source=chunk.source,
            id=chunk.id,
            page=chunk.page
        )

    def create_concepts(
        self,
        session,
        chunk_id,
        concepts
    ):

        for concept in concepts:
            session.run(
                """
                MATCH (c:Chunk {id:$chunk})
                MERGE (x:Concept {name:$concept})
                MERGE (c)-[:MENTIONS]->(x)
                """,
                chunk=chunk_id,
                concept=concept
            )

    def create_relationships(
        self,
        session,
        relationships
    ):

        for relation in relationships:
            query = f"""
            MERGE (a:Concept {{name:$source,workspace:$workspace}})
            MERGE (b:Concept {{name:$target,workspace:$workspace}})
            MERGE (a)-[:{relation['relationship']}]->(b)
            """
            session.run(
                query,
                workspace=self.workspace, 
                source=relation["source"],
                target=relation["target"]
            )

    def extract_graph(self, text):

        prompt = f"""
            Extract the important concepts and semantic relationships.
            Return ONLY valid JSON.
            Format:
            {{
                "concepts":[
                    "..."
                ],
                "relationships":[
                    {{
                        "source":"",
                        "relationship":"",
                        "target":""
                    }}
                ]
            }}
            Use relationship names like
            USES
            PART_OF
            CAUSES
            DEPENDS_ON
            IMPLEMENTS
            PRODUCES
            REQUIRES
            CONTAINS
            Text:
            {text}
            """

        response = self.llm.invoke(prompt).content

        try:
            graph = json.loads(response)
            concepts = graph.get("concepts", [])
            relationships = graph.get(
                "relationships",
                []
            )

        except Exception as e:
            print("JSON Parse Error:", e)
            print(response)
            raise

        return concepts, relationships

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