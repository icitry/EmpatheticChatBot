<script>
	import ChatInput from "./ChatInput.svelte";
	import ChatMessagesList from "./ChatMessagesList.svelte";

	let appendMessage;

	const submitMessage = async (msg) => {
		appendMessage(msg, true);

		const res = await fetch('/_api/chat', {
			method: 'POST',
			body: JSON.stringify({
				message: msg
			}),
			headers: {
				'content-type': 'application/json'
			}
		});

		if (!res.ok) {
			appendMessage(null, false);
		} else {
			const resJson = await res.json();
			appendMessage(resJson['data'], false);
		}
	};
</script>

<div class="chat_container">
	<div class="chat_header">empathetic mf</div>
	<ChatMessagesList bind:appendMessage></ChatMessagesList>
	<ChatInput onSubmit={submitMessage}></ChatInput>
</div>

<style lang="scss">
	.chat_container {
		width: 50%;
		height: 90%;
		display: flex;
		flex-flow: column;
		position: relative;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		box-shadow: 6px 6px 2px 1px var(--chat-message-bg);
		border-radius: 5px;
	}

	.chat_header {
		flex: 0 1 auto;
		width: 100%;
		height: 3rem;
		line-height: 3rem;
		text-align: center;
		border-top-left-radius: 5px;
		border-top-right-radius: 5px;
		color: var(--chat-message-text-color);
		text-shadow: 1px 1px 2px black;
		font-size: 1.25rem;
		font-weight: 700;
		background-color: var(--chat-features-bg);
	}
</style>
