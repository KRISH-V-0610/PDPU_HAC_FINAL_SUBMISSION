import axios, { AxiosError } from "axios";

// ✅ Safe Stringify Function to Prevent JSON Errors
const safeStringify = (obj: any) => {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (error) {
    return "Error: Unable to stringify object";
  }
};

// ✅ Set Up Axios Instance
const api = axios.create({
  baseURL:
  
     "https://pdpu-hac-final-submission-4.onrender.com"
    ,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});


// ✅ Request Interceptor
api.interceptors.request.use(
  (config) => {
    try {
      if (typeof window !== "undefined") {
        const token = localStorage.getItem("token");
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
      }

      console.log("API Request:", {
        url: config.url,
        method: config.method,
        headers: config.headers,
        data: config.data,
      });

      return config;
    } catch (error) {
      console.error("Request interceptor error:", error);
      return Promise.reject({
        message: "Failed to prepare request",
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  },
  (error) => {
    console.error("Request error:", error);
    return Promise.reject({
      message: "Request failed",
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
);

// ✅ Response Interceptor (Handles 404 & Auth Errors)
api.interceptors.response.use(
  (response) => {
    try {
      console.log("API Response:", {
        url: response.config.url,
        status: response.status,
        data: response.data,
      });
      return response;
    } catch (error) {
      console.error("Response success handler error:", error);
      return response;
    }
  },
  (error: AxiosError) => {
    try {
      // Create a detailed error object
      const errorDetails = {
        request: {
          url: error.config?.url || 'Unknown URL',
          method: error.config?.method || 'Unknown method',
          headers: error.config?.headers || {},
          data: error.config?.data || null,
        },
        response: {
          status: error.response?.status || 'No status',
          statusText: error.response?.statusText || 'No status text',
          data: error.response?.data || null,
        },
        message: error.message || 'No message',
        name: error.name || 'No name',
        stack: error.stack || 'No stack trace',
        isAxiosError: error.isAxiosError || false,
      };

      if (!error.config) {
        console.error("API Error: No request config found. Possible network failure.", errorDetails);
        return Promise.reject({
          message: "Request failed before reaching the server",
          error: error.message || "Unknown error",
          details: errorDetails,
        });
      }

      // ✅ Handle 404 Not Found Errors
      if (error.response?.status === 404) {
        console.warn(`API endpoint not found: ${error.config.url}`, errorDetails);
        return Promise.reject({
          message: `Endpoint ${error.config.url} does not exist`,
          status: 404,
          details: errorDetails,
        });
      }

      console.error("API Error Response:", safeStringify(errorDetails));

      // ✅ Handle Authentication Errors (401)
      if (error.response?.status === 401 && typeof window !== "undefined") {
        localStorage.removeItem("token");
        localStorage.removeItem("organization");
        window.location.href = "/auth";
      }

      // Build a comprehensive error object
      return Promise.reject({
        // message: error.response?.data?.message || error.message || "An error occurred",
        status: error.response?.status || "Unknown",
        data: error.response?.data || null,
        details: errorDetails,
      });
    } catch (e) {
      console.error("Response error handler error:", e);
      return Promise.reject({
        message: "Failed to process error response",
        error: e instanceof Error ? e.message : "Unknown error",
      });
    }
  }
);

export default api;
