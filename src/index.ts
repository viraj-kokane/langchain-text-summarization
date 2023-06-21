import * as dotenv from "dotenv";
import { OpenAI } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

dotenv.config();

const model = new OpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
});

const text = fs.readFileSync("text-samples/article.txt", "utf8");
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
const docs = await textSplitter.createDocuments([text]);

// This convenience function creates a document chain prompted to summarize a set of documents.
const chain = loadSummarizationChain(model, {
  type: "map_reduce",
  returnIntermediateSteps: true,
});
const res = await chain.call({
  input_documents: docs,
});

console.log(res);
