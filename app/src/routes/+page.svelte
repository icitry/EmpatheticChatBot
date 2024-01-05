<script>
	import ChatContainer from "../components/ChatContainer.svelte";
</script>

<div class="main">
	<ChatContainer></ChatContainer>
</div>

<style lang="scss">
    :global(body) {
        box-sizing: border-box;
        margin: 0;
    }

    .main {
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        background-color: var(--site-bg);
    }
</style>
