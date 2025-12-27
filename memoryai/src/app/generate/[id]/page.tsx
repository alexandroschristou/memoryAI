import GenerateClient from "./GenerateClient";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <GenerateClient id={id} />;
}
