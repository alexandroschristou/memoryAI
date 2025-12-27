function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export const config = {
  videoProvider: process.env.VIDEO_PROVIDER ?? "fake",

  runway: {
    apiKey: requireEnv("RUNWAY_API_KEY"),
    apiBase: process.env.RUNWAY_API_BASE ?? "https://api.runwayml.com/v1",
  },
};
