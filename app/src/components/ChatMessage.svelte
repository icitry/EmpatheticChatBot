<script>
	export let data;
</script>

{#if data}
	<div class="chat_message_container">
		<div class="chat_message_content {data['isMe'] ? 'is_me_message' : 'is_not_me_message'}">
			{#if data['msg'] === null}
				Could not load message.
			{:else if data['isMe']}
				{data['msg']}
			{:else}
                <iframe src={data['msg']} frameBorder="0" class="giphy-embed" title=""></iframe>
			{/if}
		</div>
	</div>
{/if}

<style>
	.chat_message_container {
		width: 100%;
		height: fit-content;
		margin: 1rem 0rem;
	}

	.chat_message_content {
		padding: 0.5rem;
		height: fit-content;
        width: fit-content;
		max-width: 60%;
		color: var(--chat-message-text-color);
		border-radius: 5px;
	}

	.is_not_me_message {
		border: 3px solid var(--chat-features-bg);
	}

	.is_me_message {
		margin-left: auto;
		margin-right: 0;
		background-color: var(--chat-features-bg);
	}
</style>
