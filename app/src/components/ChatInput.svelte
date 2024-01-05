<script>
	export let onSubmit;
	let msg = '';
</script>

<div class="chat_input_container">
	<form
		on:submit|preventDefault={() => {
			if (msg.length > 0) {
				onSubmit(msg);
				msg = '';
			}
		}}
	>
		<input type="text" id="chat_input" placeholder="hit me up" bind:value={msg} />
		<button class="chat_submit" id="chat_submit"><i class="material-icons">send</i></button>
	</form>
</div>

<style lang="scss">
	.chat_input_container {
		flex: 0 1 3rem;
		width: 100%;
		height: 3rem;
		background: var(--chat-bg);
		border-bottom-left-radius: 5px;
		border-bottom-right-radius: 5px;
		box-sizing: border-box;
		border: 3px solid var(--chat-features-bg);
	}

    .chat_input_container > form {
        width: 100%;
        height: 100%;
        display: flex;
		justify-content: space-between;
    }

	.chat_submit {
		position: relative;
		top: 50%;
		transform: translate(-10%, -45%);
		background: transparent;
		border: none;
		border-radius: 50%;
		color: var(--chat-features-bg);
		width: 35px;
		height: 35px;
		cursor: pointer;
	}

	#chat_input {
		background: var(--chat-bg);
		width: 90%;
		height: 2.6rem;
		border: none;
		resize: none;
		outline: none;
		color: black;
		border-top: none;
		padding: 0 10px;
		border-bottom-right-radius: 5px;
		border-bottom-left-radius: 5px;
	}
</style>
