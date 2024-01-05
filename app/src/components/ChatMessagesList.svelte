<script>
	import ChatMessage from "./ChatMessage.svelte";

	export const appendMessage = (msg, isMe) => {
		messagesList = [
			...messagesList,
			{
				msg: msg,
				isMe: isMe
			}
		];
	};

	let messagesList = [];
</script>

<div class="chat_messages_list_svelte_container">
	{#each messagesList as message}
		<ChatMessage data={message}></ChatMessage>
	{/each}
</div>

<style lang="scss">
	.chat_messages_list_svelte_container {
		flex: 1 1 auto;
		width: 100%;
		height: max-content;
		overflow-y: scroll;
		background-color: var(--chat-bg);
	}

	.chat_messages_list_svelte_container::-webkit-scrollbar-track {
		-webkit-box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
		background-color: var(--scrollbar-bg);
	}

	.chat_messages_list_svelte_container::-webkit-scrollbar {
		width: 5px;
		background-color: var(--scrollbar-bg);
	}

	.chat_messages_list_svelte_container::-webkit-scrollbar-thumb {
		background-color: var(--chat-features-bg);
	}
</style>
