import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const apiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({
  apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

async function startServer() {
  const app = express();
  const PORT = parseInt(process.env.PORT || '3000', 10);

  // Parse JSON bodies with support for images/audio attachments (up to 25MB)
  app.use(express.json({ limit: '25mb' }));

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(apiKey),
      defaultModel: 'gemini-3.8-flash',
    });
  });

  // Streaming Chat API via SSE
  app.post('/api/chat/stream', async (req, res) => {
    // Set headers for Server-Sent Events
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    const sendEvent = (data: Record<string, unknown>) => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    if (!apiKey) {
      sendEvent({
        type: 'error',
        message: 'GEMINI_API_KEY is not configured on the server. Please check the Secrets panel.',
      });
      res.end();
      return;
    }

    try {
      const {
        messages = [],
        model = 'gemini-3.8-flash',
        systemInstruction,
        useSearch = false,
        thinkingLevel,
        temperature,
      } = req.body;

      if (!Array.isArray(messages) || messages.length === 0) {
        sendEvent({ type: 'error', message: 'No messages provided.' });
        res.end();
        return;
      }

      // Convert messages to Gemini API contents format
      // Rules:
      // - roles must be 'user' or 'model'
      // - each content has parts
      const contents = messages.map((msg: any) => {
        const parts: any[] = [];

        // Add attachments if any (multimodal: image, audio, etc.)
        if (Array.isArray(msg.attachments)) {
          for (const att of msg.attachments) {
            if (att.data && att.mimeType) {
              parts.push({
                inlineData: {
                  mimeType: att.mimeType,
                  data: att.data,
                },
              });
            }
          }
        }

        // Add text part
        if (msg.content !== undefined && msg.content !== null) {
          parts.push({
            text: String(msg.content),
          });
        }

        return {
          role: msg.role === 'assistant' || msg.role === 'model' ? 'model' : 'user',
          parts: parts.length > 0 ? parts : [{ text: '' }],
        };
      });

      // Gemini configuration
      const config: any = {};

      if (systemInstruction && typeof systemInstruction === 'string' && systemInstruction.trim()) {
        config.systemInstruction = systemInstruction.trim();
      }

      if (typeof temperature === 'number') {
        config.temperature = temperature;
      }

      // Grounding with Google Search if requested
      if (useSearch) {
        config.tools = [{ googleSearch: {} }];
      }

      // Thinking level if specified for gemini-3 series
      if (thinkingLevel && thinkingLevel !== 'NONE') {
        if (thinkingLevel === 'LOW') {
          config.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
        } else if (thinkingLevel === 'MEDIUM') {
          config.thinkingConfig = { thinkingLevel: ThinkingLevel.MEDIUM };
        } else if (thinkingLevel === 'HIGH') {
          config.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
        }
      }

      const startTime = Date.now();
      let stream;
      let usedModel = model || 'gemini-3.8-flash';

      try {
        stream = await ai.models.generateContentStream({
          model: usedModel,
          contents,
          config,
        });
      } catch (initialErr: any) {
        const errMsg = String(initialErr?.message || '');
        // If the primary model is overloaded (503 / RESOURCE_EXHAUSTED), fallback to gemini-3.1-flash-lite
        if (errMsg.includes('503') || errMsg.includes('overloaded') || errMsg.includes('RESOURCE_EXHAUSTED')) {
          console.warn(`Primary model ${usedModel} overloaded, falling back to gemini-3.1-flash-lite`);
          usedModel = 'gemini-3.1-flash-lite';
          // flash-lite doesn't support HIGH thinking level, minimal by default
          const fallbackConfig = { ...config };
          if (fallbackConfig.thinkingConfig) {
            delete fallbackConfig.thinkingConfig;
          }
          stream = await ai.models.generateContentStream({
            model: usedModel,
            contents,
            config: fallbackConfig,
          });
        } else {
          throw initialErr;
        }
      }

      let fullText = '';
      const groundingSources: Array<{ title: string; url: string }> = [];

      for await (const chunk of stream) {
        const chunkText = chunk.text;
        if (chunkText) {
          fullText += chunkText;
          sendEvent({
            type: 'chunk',
            text: chunkText,
          });
        }

        // Extract grounding web search results if available
        const metadata = chunk.candidates?.[0]?.groundingMetadata;
        if (metadata?.groundingChunks) {
          for (const gChunk of metadata.groundingChunks as any[]) {
            if (gChunk.web?.uri) {
              const url = gChunk.web.uri;
              const title = gChunk.web.title || new URL(url).hostname;
              if (!groundingSources.some((s) => s.url === url)) {
                groundingSources.push({ title, url });
              }
            }
          }
        }
      }

      // Send grounding sources if collected
      if (groundingSources.length > 0) {
        sendEvent({
          type: 'grounding',
          sources: groundingSources,
        });
      }

      const durationMs = Date.now() - startTime;
      sendEvent({
        type: 'done',
        stats: {
          durationMs,
          charCount: fullText.length,
        },
      });
      res.end();
    } catch (error: any) {
      console.error('Error in /api/chat/stream:', error);
      sendEvent({
        type: 'error',
        message: error?.message || 'An unexpected error occurred while communicating with Gemini.',
      });
      res.end();
    }
  });

  // Text-To-Speech endpoint using gemini-3.8-flash-lite-tts
  app.post('/api/chat/tts', async (req, res) => {
    try {
      if (!apiKey) {
        res.status(500).json({ error: 'GEMINI_API_KEY is not configured on the server.' });
        return;
      }

      const { text, voiceName = 'Zephyr' } = req.body;
      if (!text || typeof text !== 'string') {
        res.status(400).json({ error: 'Missing or invalid text parameter' });
        return;
      }

      // Truncate to reasonable length for TTS synthesis if overly long
      const textToSpeak = text.slice(0, 1500).replace(/[*#`_~[\]()]/g, '');

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash-lite-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: textToSpeak,
                speechMetadata: {
                  style: 'Clear, engaging, articulate natural assistant tone',
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: voiceName || 'Zephyr' },
            },
          },
        },
      });

      const audioBase64 = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!audioBase64) {
        res.status(500).json({ error: 'Failed to synthesize audio from Gemini TTS.' });
        return;
      }

      res.json({
        audioBase64,
        mimeType: 'audio/wav',
      });
    } catch (error: any) {
      console.error('TTS error:', error);
      res.status(500).json({ error: error?.message || 'Failed to synthesize audio.' });
    }
  });

  // Audio transcription endpoint using gemini-3.5-transcribe
  app.post('/api/chat/transcribe', async (req, res) => {
    try {
      if (!apiKey) {
        res.status(500).json({ error: 'GEMINI_API_KEY is not configured on the server.' });
        return;
      }

      const { audioBase64, mimeType = 'audio/webm' } = req.body;
      if (!audioBase64) {
        res.status(400).json({ error: 'Missing audioBase64' });
        return;
      }

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: audioBase64,
              },
            },
            {
              text: 'Transcribe this spoken speech accurately. Output only the transcribed text, without timestamps, labels, or additional commentary.',
            },
          ],
        },
      });

      const transcription = response.text?.trim() || '';
      res.json({ text: transcription });
    } catch (error: any) {
      console.error('Transcription error:', error);
      res.status(500).json({ error: error?.message || 'Failed to transcribe audio' });
    }
  });

  // In production, serve the built Vite dist; in development, mount Vite middleware
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`NexusAI Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
