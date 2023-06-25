import { PrismaClient } from '@prisma/client';
import { getServerSession } from '$app/env';

let prismaClient: PrismaClient | null = null;

export function getPrismaClient() {
	if (getServerSession()) {
		if (!prismaClient) {
			prismaClient = new PrismaClient();
		}
		return prismaClient;
	}
}
