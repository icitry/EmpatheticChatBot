import { API_BASE_URL } from '../../../defs';

/** @type {import('./$types').RequestHandler} */
export const POST = async (request) => {
    const url = `${API_BASE_URL}/chat`
    const reqBody = await request.request.json();

    if (!('message' in reqBody))
        return new Response(JSON.stringify({
            error: 'No message provided.'
        }),
            {
                status: 400
            })

    const res = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'message': reqBody['message']
        })
    });

    let res_data = await res.json();

    if (!res.ok)
        return new Response(JSON.stringify({
            error: res_data['error']
        }),
            {
                status: res.status
            })

    return new Response(JSON.stringify(
        {
            "data": res_data["data"]
        }
    ));
}