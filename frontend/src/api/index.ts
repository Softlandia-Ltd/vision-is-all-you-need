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

const initRequest = (args: fetcherArgs<any>) => {
  const { method, endpoint, body } = args;

  let headers: HeadersInit = {
    accept: "application/json",
  };

  if (!(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const url = backendUrl + endpoint;

  const request = new Request(url, {
    method,
    headers,
    body: body && !(body instanceof FormData) ? JSON.stringify(body) : body,
  });

  return request;
};

const fetchStream = async <C, T>(args: eventSourceFetcherArgs<C, T>) => {
  const request = initRequest(args);
  const headers: Record<string, string> = {};
  request.headers.forEach((value, key) => {
    headers[key] = value;
  });

  const res = await fetchEventSource(request, {
    headers,
    openWhenHidden: true,
    async onopen(response) {
      if (response.ok && response.status === 200) {
        args.onOpen && args.onOpen();
      } else if (
        response.status >= 400 &&
        response.status < 500 &&
        response.status !== 429
      ) {
        console.error("Error connecting to server", response.statusText);
        args.onError && args.onError(new Error(response.statusText));
      }
    },
    onmessage(msg) {
      console.log(msg);
      try {
        args.onMessage(JSON.parse(msg.data) as T, msg.event);
      } catch {
        args.onMessage(msg.data as T, msg.event);
      }
      if (msg.event === "complete") {
        console.log("Stream complete");
      }
    },
    onclose() {
      console.log("Connection closed by the server");
      args.onClose && args.onClose();
    },
    onerror(err) {
      console.log("There was an error from server", err);
      args.onError && args.onError(err);
    },
  });

  return res;
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
