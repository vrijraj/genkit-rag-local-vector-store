// Import required dependencies and types
import { z, genkit, indexerRef, run } from "genkit";
import { Document } from "genkit/retriever";
import {
  googleAI,
  gemini15Flash,
  textEmbeddingGecko001,
} from "@genkit-ai/googleai";
import pdf from "pdf-parse";
import fs from "fs";
import {
  devLocalIndexerRef,
  devLocalVectorstore,
  devLocalRetrieverRef,
} from "@genkit-ai/dev-local-vectorstore";
import { chunk } from "llm-chunk";

// Load environment variables from .env file
import dotenv from "dotenv";
dotenv.config();

// Initialize Genkit with Google AI configuration
// Set up the AI instance with necessary plugins and model
const ai = genkit({
  plugins: [
    // Configure Google AI with API key from environment variables
    googleAI({ apiKey: process.env.GOOGLE_GENAI_API_KEY }),
    // Set up local vector store for document indexing
    devLocalVectorstore([
      {
        embedder: textEmbeddingGecko001, // Use Google's text embedding model
        indexName: "facts", // Name the index for storing document vectors
      },
    ]),
  ],
  model: gemini15Flash, // Use Gemini 1.5 Flash as the primary model
});

// Define the indexer reference for PDF processing
export const PdfIndexer = devLocalIndexerRef("facts");

// Configure text chunking parameters
const chunkingConfig = {
  minLength: 1000,    // Minimum chunk size in characters
  maxLength: 2000,    // Maximum chunk size in characters
  splitter: "sentence", // Split text at sentence boundaries
  overlap: 100,       // Number of overlapping characters between chunks
  delimiters: "",     // No additional delimiters
} as any;

// Define the indexing flow for processing PDF documents
const indexMenu = ai.defineFlow(
  {
    name: "indexMenu",
    inputSchema: z.string(), // Accept string input
  },
  async () => {
    // Read and parse the PDF file
    const pdfTxt = await pdf(fs.readFileSync("Avengers.pdf"));

    // Split the PDF text into manageable chunks
    const chunks = await run("chunk-it", async () =>
      chunk(pdfTxt.text, chunkingConfig)
    );

    // Index each chunk as a separate document
    await ai.index({
      indexer: PdfIndexer,
      documents: chunks.map((c: string) => Document.fromText(c)),
    });
  }
);

// Create a retriever reference for querying the indexed documents
export const retriever = devLocalRetrieverRef("facts");

// Define the main query flow for handling user questions
const helloFlow = ai.defineFlow(
  { 
    name: "MainMenu", 
    inputSchema: z.string(),    // Accept string input (question)
    outputSchema: z.string()    // Return string output (answer)
  },
  async (ques) => {
    // Retrieve relevant documents based on the question
    const docs = await ai.retrieve({
      retriever: retriever,
      query: ques,
      options: { k: 3 },  // Retrieve top 3 most relevant documents
    });

    // Generate response using the question and retrieved documents
    const { text } = await ai.generate({
      prompt: ques,
      docs,
    });
    return text;
  }
);

// Commented out execution code
// (async () => {
//   console.log(await helloFlow("Hello Gemini"));
// })();