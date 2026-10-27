import type { CoreMessage } from "./types.js";

/**
 * Adjust compression range boundaries to keep a `reasoning` message together
 * with the assistant text/tool-call it belongs to.
 *
 * Reasoning models (DeepSeek-R1, GLM-4.6 thinking, Qwen-QwQ, Anthropic
 * thinking) emit a `reasoning_content` / thinking block that strict providers
 * require to be echoed back alongside the response on every subsequent
 * request. In acp-kernel that block is a separate `contentType: "reasoning"`
 * message immediately preceding the assistant text/tool-call of the same turn.
 * If a compression range covers only one half of the pair, the rebuilt
 * conversation ships reasoning without its response (or vice versa) and the
 * provider returns HTTP 400 (DeepSeek: "reasoning_content in the thinking mode
 * must be passed back to the API").
 *
 * This is the reasoning analogue of {@link adjustBoundariesForToolPairs}:
 * before a range is applied, pull the orphan half INTO the range so the pair
 * compresses together — zero information loss. Only MESSAGE-boundary ranges
 * are adjusted (block-boundary ranges are left untouched, like tool pairs).
 *
 * Pairing is adjacency-based — there is no shared id (unlike toolCallId). A
 * `reasoning` message pairs with the assistant text/tool-call immediately
 * following its reasoning run, and an assistant text/tool-call pairs with the
 * reasoning run immediately preceding it. This matches the round-trip contract
 * every adapter relies on when reconstructing reasoning_content.
 *
 * @returns Adjusted { startIndex, endIndex } — may be wider than input.
 */
export function adjustBoundariesForReasoningPairs(
  startIndex: number,
  endIndex: number,
  messages: CoreMessage[],
): { startIndex: number; endIndex: number } {
  if (startIndex > endIndex) {
    return { startIndex, endIndex };
  }
  let newStartIndex = startIndex;
  let newEndIndex = endIndex;

  for (let i = startIndex; i <= endIndex && i < messages.length; i++) {
    const msg = messages[i];
    if (!msg) continue;

    if (msg.contentType === "reasoning") {
      // Forward: pull the companion assistant text/tool-call that follows
      // this reasoning run into the range.
      let j = i;
      while (
        j + 1 < messages.length &&
        messages[j + 1]!.contentType === "reasoning"
      ) {
        j++;
      }
      // Prior compression may insert block summaries between the reasoning
      // run and its assistant companion. Treat these summaries as transparent.
      let companionIndex = j + 1;
      while (
        companionIndex < messages.length &&
        messages[companionIndex]!.role === "system" &&
        typeof messages[companionIndex]!.id === "string" &&
        messages[companionIndex]!.id.startsWith("acp_summary_")
      ) {
        companionIndex++;
      }
      const companion = messages[companionIndex];
      if (
        companion !== undefined &&
        companion.role === "assistant" &&
        (companion.contentType === "text" ||
          companion.contentType === "tool-call")
      ) {
        // Pull the WHOLE assistant burst following the reasoning run, not just
        // the first companion (#684): a turn may carry several tool-calls, and
        // pulling one leaves its siblings outside the range. The composed
        // fixpoint's tool-pair pass then widens for every result.
        let e = companionIndex;
        while (
          e + 1 < messages.length &&
          messages[e + 1]!.role === "assistant" &&
          (messages[e + 1]!.contentType === "text" ||
            messages[e + 1]!.contentType === "tool-call")
        ) {
          e++;
        }
        if (e > newEndIndex) newEndIndex = e;
      }
    }

    if (
      msg.role === "assistant" &&
      (msg.contentType === "text" || msg.contentType === "tool-call")
    ) {
      // Backward: pull the reasoning run immediately preceding this assistant
      // message into the range.
      let k = i - 1;
      let runStart = i;
      while (k >= 0) {
        if (messages[k]!.contentType === "reasoning") {
          runStart = k;
          k--;
          continue;
        }
        // Block summaries inserted by an earlier compression do not break
        // the logical adjacency between reasoning and its companion.
        if (
          messages[k]!.role === "system" &&
          typeof messages[k]!.id === "string" &&
          messages[k]!.id.startsWith("acp_summary_")
        ) {
          k--;
          continue;
        }
        break;
      }
      if (
        runStart < i &&
        runStart >= 0 &&
        messages[runStart]!.contentType === "reasoning" &&
        runStart < newStartIndex
      ) {
        newStartIndex = runStart;
      }
    }
  }

  return { startIndex: newStartIndex, endIndex: newEndIndex };
}
