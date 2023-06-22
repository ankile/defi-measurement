import dotenv from 'dotenv';
import { MongoClient } from 'mongodb';

dotenv.config();

const uri = process.env.MONGODB_URI;

if (!uri) {
	throw new Error('Please add your Mongo URI to .env.local');
}

const client = new MongoClient(uri);
const clientPromise = client.connect();

// Export a module-scoped MongoClient promise.
// By doing this in a separate module,
// the client can be shared across functions.
export default clientPromise;
