import { fetchEventSource } from "@microsoft/fetch-event-source";
const backendUrl = import.meta.env.VITE_BACKEND_URL;

type fetcherArgs<T> = {
  method: string;
  endpoint: string;
  body?: T | FormData;
  stream?: boolean;
};

type eventSourceArgs<T> = {
  onMessage: (data: T, event: string) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Error) => void;
};

type eventSourceFetcherArgs<C, T> = fetcherArgs<C> & eventSourceArgs<T>;

const initRequest = (args: fetcherArgs<unknown>) => {
  const { method, endpoint, body } = args;

  const headers: Record<string, string> = {
    accept: "application/json",
  };

  if (!(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const url = backendUrl + endpoint;

  const request = new Request(url, {
    method,
    headers,
    body:
      body && !(body instanceof FormData)
        ? JSON.stringify(body)
        : (body as FormData | undefined),
  });

  return request;
};

/** Thrown to end a stream the server closed on purpose (not a failure). */
class StreamClosed extends Error {}

const fetchStream = async <C, T>(args: eventSourceFetcherArgs<C, T>) => {
  const request = initRequest(args);
  const headers: Record<string, string> = {};
  request.headers.forEach((value, key) => {
    headers[key] = value;
  });

  let reported = false;
  const fail = (err: unknown) => {
    if (reported) return;
    reported = true;
    args.onError?.(err instanceof Error ? err : new Error(String(err)));
  };

  try {
    await fetchEventSource(request, {
      headers,
      openWhenHidden: true,
      async onopen(response) {
        if (response.ok && response.status === 200) {
          args.onOpen?.();
          return;
        }
        const body = await response.text().catch(() => "");
        throw new Error(
          `Request failed (${response.status} ${response.statusText})` +
            (body ? `: ${body.slice(0, 300)}` : "")
        );
      },
      onmessage(msg) {
        try {
          args.onMessage(JSON.parse(msg.data) as T, msg.event);
        } catch {
          args.onMessage(msg.data as T, msg.event);
        }
      },
      onclose() {
        args.onClose?.();
        // The library retries whenever the connection ends unless we throw,
        // which would silently re-run the whole request.
        throw new StreamClosed();
      },
      onerror(err) {
        // Rethrowing marks the error fatal; returning would retry forever.
        throw err;
      },
    });
  } catch (err) {
    if (!(err instanceof StreamClosed)) {
      fail(err);
    }
  }
};

export const postStream = async <C, T>(
  endpoint: string,
  body: C,
  onMessage: (data: T, event: string) => void,
  onOpen?: () => void,
  onClose?: () => void,
  onError?: (err: Error) => void
) =>
  await fetchStream<C, T>({
    method: "POST",
    endpoint,
    body,
    onMessage,
    onOpen,
    onClose,
    onError,
  });

export const postFilesStream = async <T>(
  endpoint: string,
  body: FormData,
  onMessage: (data: T, event: string) => void,
  onOpen?: () => void,
  onClose?: () => void,
  onError?: (err: Error) => void
) => {
  await fetchStream<FormData, T>({
    method: "POST",
    endpoint,
    body,
    onMessage,
    onOpen,
    onClose,
    onError,
  });
};
